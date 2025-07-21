#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import librosa
import torchaudio
import scipy.io.wavfile
import scipy.signal
from typing import Optional, Union, Dict, Any, Tuple
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
random.seed(0)
from phonemizer import phonemize
from phonemizer.separator import Separator
import numpy as np
# load packages
import time
import random
import yaml
import scipy.signal
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import re
from indic_numtowords import num2words
import sys 
sys.path.append('/home/user/voice/StyleTTS2')
from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
# %matplotlib inline

class TTSAgent:
    """
    Text-to-Speech Agent that converts text to audio using the English TTS model.
    
    This class wraps the functionality from the TTS inference script into a reusable
    component that can be integrated into various applications.
    """
    
    def __init__(self, 
                model_path: str = "/home/user/voice/StyleTTS2/Models/indic_voices/epoch_2nd_00049.pth",
                config_path: str = "/home/user/voice/StyleTTS2/Models/indic_voices/config_ft.yml",
                device: Optional[str] = "cuda"):
        """
        Initialize the TTS Agent with model paths and configuration.
        
        Args:
            model_path: Path to the TTS model checkpoint
            config_path: Path to the model configuration YAML file
            device: Device to run inference on ('cuda' or 'cpu'). If None, will auto-detect.
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing TTS Agent using device: {self.device}")
        
        # Load dependencies
        self._load_dependencies()
        
        # Load and initialize the model
        self._initialize_model()
        
        # Setup mel spectrogram transform
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mel_mean, self.mel_std = -4, 4
        
        # Initialize text cleaner
        self._initialize_text_cleaner()
        
        print("TTS Agent initialized successfully")
        
    def _load_dependencies(self):
        """Load necessary dependencies for the TTS model."""
        try:
            import yaml
            from munch import Munch
            
            # These would be imported from the original script
            from text_utils import TextCleaner
            from models import build_model, load_ASR_models, load_F0_models
            from utils import recursive_munch
            from Utils.PLBERT.util import load_plbert
            # from Utils.phonemize.english_phonemizer import english_phonemize
            
            # Store these imports as instance attributes
            self.yaml = yaml
            self.Munch = Munch
            self.TextCleaner = TextCleaner
            self.build_model = build_model
            self.load_ASR_models = load_ASR_models
            self.load_F0_models = load_F0_models
            self.recursive_munch = recursive_munch
            self.load_plbert = load_plbert
            # self.english_phonemize = english_phonemize
            
        except ImportError as e:
            raise ImportError(f"Failed to import dependencies: {e}. Please make sure all required packages are installed.")
    
    def _initialize_model(self):
        """Initialize and load the TTS model."""
        try:
            # Set random seeds for reproducibility
            torch.manual_seed(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            np.random.seed(0)
            
            # Load configuration
            config = self.yaml.safe_load(open(self.config_path))
            model_params = self.recursive_munch(config['model_params'])
            
            # Load pretrained ASR model
            ASR_config = config.get('ASR_config', False)
            ASR_path = config.get('ASR_path', False)
            text_aligner = self.load_ASR_models(ASR_path, ASR_config)
            print("Text aligner loaded")
            
            # Load pretrained F0 model
            F0_path = config.get('F0_path', False)
            pitch_extractor = self.load_F0_models(F0_path)
            print("Pitch extractor loaded")
            
            # Load BERT model
            BERT_path = config.get('PLBERT_dir', False)
            plbert = self.load_plbert(BERT_path)
            print("BERT model loaded")
            
            # Build model
            self.model = self.build_model(model_params, text_aligner, pitch_extractor, plbert)
            _ = [self.model[key].eval() for key in self.model]
            _ = [self.model[key].to(self.device) for key in self.model]
            
            # Load model parameters
            params_whole = torch.load(self.model_path, map_location='cpu')
            params = params_whole['net']
            
            for key in self.model:
                if key in params:
                    print(f'{key} loaded')
                    try:
                        self.model[key].load_state_dict(params[key])
                    except:
                        from collections import OrderedDict
                        state_dict = params[key]
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        # load params
                        self.model[key].load_state_dict(new_state_dict, strict=False)
            
            _ = [self.model[key].eval() for key in self.model]
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    def _initialize_text_cleaner(self):
        """Initialize the text cleaner component."""
        self.textcleaner = self.TextCleaner()
    
    def _length_to_mask(self, lengths):
        """Convert lengths to mask."""
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
    def _preprocess(self, wave):
        """Preprocess audio waveform to mel spectrogram."""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mel_mean) / self.mel_std
        return mel_tensor
    
    def get_style_vector(self, wav_path: str) -> Optional[torch.Tensor]:
        """
        Extract style vector from reference audio.
        
        Args:
            wav_path: Path to the reference audio file
            
        Returns:
            torch.Tensor: Style vector or None if extraction fails
        """
        try:
            wave, sr = librosa.load(wav_path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            
            mel_tensor = self._preprocess(audio).to(self.device)
            
            with torch.no_grad():
                ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
                ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
            
            return torch.cat([ref_s, ref_p], dim=1)
        except Exception as e:
            print(f"Error extracting style vector: {e}")
            return None
    def clean_phonemize(
        text: str,
        language: str = "hi",
        backend: str = "espeak",
        njobs: int = 4
    ) -> str:
        """
        Phonemize a mixed Hindi-English string into a plain IPA-like string,
        removing any language-switch tags like (en) or (hi).

        Args:
            text (str): Input sentence (e.g., "CRM_SALES AI असिस्टेड कॉल लॉग Received.")
            language (str): Main language code for phonemizer (default 'hi').
            backend (str): Phonemizer backend to use (default 'espeak').
            njobs (int): Number of parallel jobs (default 4).

        Returns:
            str: Phoneme transcription without language-switch flags.
        """
        # 1. Configure separators: no delimiter for phones, a single space for words
        separator = Separator(phone="", word=" ", syllable="")

        # 2. Invoke high-level phonemize with remove-flags policy
        phoneme_list = phonemize(
            [text],
            language=language,
            backend=backend,
            separator=separator,
            strip=True,                   # trim leading/trailing whitespace
            preserve_punctuation=True,    # keep punctuation intact
            njobs=njobs,
            language_switch="remove-flags"  # strip out (en)/(hi) tags :contentReference[oaicite:0]{index=0}
        )

        # 3. Return the first (and only) phoneme string
        return phoneme_list
    def normalize_text(text: str) -> str:
        """
        Normalizes currency amounts (e.g., "1500 Rs"), 4-digit years (e.g., "2025"),
        and phone numbers (e.g., "9876543210") in the input text to their spoken Hindi equivalents.
        """

        # Pattern to identify currency amounts followed by 'Rs'
        currency_pattern = re.compile(r"(\d+)\s*Rs")

        # Pattern to identify 4-digit years
        year_pattern = re.compile(r"\b(\d{4})\b")

        # Pattern to identify phone numbers (7 or more digits, with optional separators)
        phone_pattern = re.compile(r"(\+?\d[\d\s\-]{6,}\d)")

        # Function to replace currency amounts with Hindi words
        def replace_currency(match):
            amount = int(match.group(1))
            hindi_amount = num2words(amount, lang='hi')
            return f"{hindi_amount} रुपये"

        # Function to replace years with Hindi words
        def replace_year(match):
            year = int(match.group(1))
            return num2words(year, lang='hi')

        # Function to replace phone numbers with Hindi spoken digits
        def replace_phone(match):
            number = match.group(1)
            # Remove non-digit characters
            digits = re.sub(r'\D', '', number)
            # Convert each digit to its Hindi word
            hindi_digits = [num2words(int(d), lang='hi') for d in digits]
            return ' '.join(hindi_digits)

        # Apply phone number normalization
        text = phone_pattern.sub(replace_phone, text)

        # Apply currency normalization
        text = currency_pattern.sub(replace_currency, text)

        # Apply year normalization
        text = year_pattern.sub(replace_year, text)

        return text
    def synthesize(self, 
                  text: str, 
                  reference_audio: str,
                  alpha: float = 0.3, 
                  beta: float = 0.7, 
                  diffusion_steps: int = 0, 
                  embedding_scale: float = 1.0, 
                  rate_of_speech: float = 1.0) -> Optional[np.ndarray]:
        """
        Synthesize speech from text using the TTS model.
        
        Args:
            text: Input text to convert to speech
            reference_audio: Path to reference audio file for style extraction
            alpha: Weight for reference style (0-1)
            beta: Weight for content style (0-1)
            diffusion_steps: Number of diffusion steps (not used)
            embedding_scale: Scale for embedding
            rate_of_speech: Controls the speed of speech (0.5-2.0)
            
        Returns:
            numpy.ndarray: Audio waveform at 24kHz sample rate or None if synthesis fails
        """
        try:
            text = text.strip()
            normalized_text = self.normalize_text(text)
            # Convert text to phonemes

            phonemes = self.clean_phonemize(text)
            # print(f"Phoneme length: {len(phonemes)}")
            
            # # Ensure text is not too long
            # if len(phonemes) > 300:
            #     print("Warning: Text is too long, truncating...")
            #     phonemes = phonemes[:300]
            phonemes = word_tokenize(phonemes[0])
            phonemes = ' '.join(phonemes)
            # Get style vector from reference audio
            ref_s = self.get_style_vector(reference_audio)
            if ref_s is None:
                raise ValueError("Failed to extract style vector from reference audio")
            
            # Generate speech
            wav = self._inference(
                text=phonemes,
                ref_s=ref_s,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                rate_of_speech=rate_of_speech
            )
            sr = 24000
            return sr, wav
            
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _inference(self, 
                  text: str, 
                  ref_s: torch.Tensor, 
                  alpha: float = 0.3, 
                  beta: float = 0.7, 
                  diffusion_steps: int = 0, 
                  embedding_scale: float = 1.0, 
                  rate_of_speech: float = 1.0) -> np.ndarray:
        """
        Generate speech from phonemes using the trained model.
        
        Args:
            text: Input phonemes
            ref_s: Reference style vector
            alpha: Weight for reference style
            beta: Weight for content style
            diffusion_steps: Number of diffusion steps (not used)
            embedding_scale: Scale for embedding
            rate_of_speech: Controls the speed of speech
        
        Returns:
            numpy.ndarray: Generated audio waveform
        """
        # Prepare tokens
        tokens = self.textcleaner(text)
        tokens.insert(0, 0)
        tokens.append(0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 
            sampler = DiffusionSampler(
                        self.model.diffusion.diffusion,
                        sampler=ADPM2Sampler(),
                        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
                        clamp=False
                    )
            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, 
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            # if torch.isnan(pred_dur).any():
                #print("!!! NaN found in pred_dur !!!")
                # Decide how to handle
                # return None # Example: Stop processing

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            wav_out = out.squeeze().cpu().numpy() # weird pulse at the end of the model, need to be fixed later
        
        # Apply minimal post-processing to avoid muffled sound
        try:
            # Normalize the output (but not too aggressively)
            wav_out = wav_out / (np.max(np.abs(wav_out)) + 1e-7) * 0.9
            
            # Remove DC offset
            wav_out = wav_out - np.mean(wav_out)
            
            # Apply a very gentle high-pass filter just to remove sub-bass rumble
            b, a = scipy.signal.butter(2, 40/(24000/2), 'highpass')
            wav_out = scipy.signal.filtfilt(b, a, wav_out)
        except Exception as e:
            print(f"Warning: Error during audio post-processing: {e}")
        
            
        # return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        return wav_out[..., :-100]
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 24000) -> bool:
        """
        Save generated audio to a WAV file.
        
        Args:
            audio: Audio data as numpy array
            output_path: Path to save the output file
            sample_rate: Sample rate of the audio (default: 24000)
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            scipy.io.wavfile.write(output_path, sample_rate, audio.astype(np.float32))
            print(f"Audio saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    # def stream_tts(self, text: str, reference_audio: str, 
    #               chunk_size: int = 8192, 
    #               **kwargs) -> np.ndarray:
    #     """
    #     Stream TTS output in chunks (generator function).
        
    #     Args:
    #         text: Input text to convert to speech
    #         reference_audio: Path to reference audio file for style extraction
    #         chunk_size: Size of audio chunks to yield
    #         **kwargs: Additional parameters for speech synthesis
            
    #     Yields:
    #         numpy.ndarray: Chunks of audio data
    #     """
    #     audio = self.synthesize(text, reference_audio, **kwargs)
        
    #     if audio is not None:
    #         # Yield audio in chunks
    #         for i in range(0, len(audio), chunk_size):
    #             yield audio[i:i+chunk_size]
    #     else:
    #         # Yield an empty array if synthesis failed
    #         yield np.zeros(chunk_size, dtype=np.float32)


# Simple usage example
def example_usage():
    # Initialize the TTS agent
    tts = TTSAgent(
        model_path="/home/user/voice/StyleTTS2/Models/indic_voices/epoch_2nd_00049.pth",
        config_path="/home/user/voice/StyleTTS2/Models/indic_voices/config_ft.yml"
    )
    
    # Synthesize speech
    audio = tts.synthesize(
        text="जब मैंने, सुबह की ठंडी हवा में अपने घर की बालकनी से सूरज की पहली किरणों को धरती पर बिखरते हुए देखा, तो मन एक, अनोखी ऊर्जा और शांति से भर गया.",
        reference_audio="/home/user/voice/StyleTTS2/Demo/reference/recorded.wav",
        rate_of_speech=1.0
    )
    
    # Save the audio
    if audio is not None:
        tts.save_audio(audio, "output.wav")


# Example function for integration with other systems
def tts_agent_function(text: str, reference_audio: str, output_path: str = "output.wav") -> str:
    """
    Function wrapper for TTS agent that can be easily called from other applications.
    
    Args:
        text: Text to convert to speech
        reference_audio: Path to reference audio
        output_path: Path to save the output file
        
    Returns:
        str: Path to the output file or error message
    """
    try:
        # Initialize TTS agent
        tts = TTSAgent()
        
        # Generate speech
        sr,audio = tts.synthesize(text, reference_audio)
        
        if audio is not None:
            # Save the audio
            tts.save_audio(audio, output_path)
            return output_path
        else:
            return "Failed to generate speech"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


if __name__ == "__main__":
    example_usage()