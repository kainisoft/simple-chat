"""
Optimized Chatbot Service for AMD Ryzen 9 8945HX + NVIDIA RTX 5070
High-performance inference with GPU acceleration and memory optimization
"""

import time
import logging
import asyncio
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
import psutil
import GPUtil
from .config import settings
from .database_service import database_service

logger = logging.getLogger(__name__)


class OptimizedChatbotService:
    """High-performance chatbot service optimized for the hardware setup"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.generation_config = None
        self._setup_device()
        
    def _setup_device(self):
        """Setup optimal device configuration for RTX 5070"""
        if settings.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            # Optimize CUDA settings for RTX 5070
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Set memory fraction to prevent OOM
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(settings.gpu_memory_fraction)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            # Optimize CPU settings for Ryzen 9 8945HX
            torch.set_num_threads(settings.torch_threads)
            logger.info(f"Using CPU with {settings.torch_threads} threads")
    
    async def load_model(self):
        """Load and optimize the model for inference"""
        try:
            logger.info(f"Loading model: {settings.chatbot_model_name}")
            start_time = time.time()

            # Configure quantization for better performance on RTX 5070
            quantization_config = None
            if self.device.type == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False,
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.chatbot_model_name,
                cache_dir=settings.chatbot_model_cache_dir,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            model_kwargs = {
                "cache_dir": settings.chatbot_model_cache_dir,
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.chatbot_model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not model_kwargs.get("device_map"):
                self.model = self.model.to(self.device)
            
            # Enable optimizations
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_length=settings.max_length,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            return False

    async def generate_response(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate optimized response using GPU acceleration"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()
        gpu_used = self.device.type == "cuda"

        try:
            # Prepare input with conversation context
            input_text = await self._prepare_input(message, conversation_id)

            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.max_length // 2
            ).to(self.device)

            # Override generation parameters if provided
            gen_config = self.generation_config
            if temperature is not None or max_length is not None:
                gen_config = GenerationConfig(
                    max_length=max_length or settings.max_length,
                    temperature=temperature or settings.temperature,
                    top_p=settings.top_p,
                    top_k=settings.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Generate response with optimizations
            with torch.no_grad():
                if gpu_used:
                    # Use CUDA optimizations
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            inputs,
                            generation_config=gen_config,
                            attention_mask=torch.ones_like(inputs),
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    # CPU generation
                    outputs = self.model.generate(
                        inputs,
                        generation_config=gen_config,
                        attention_mask=torch.ones_like(inputs),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Clean up response
            response = self._clean_response(response)

            processing_time = time.time() - start_time

            return {
                "response": response,
                "processing_time": processing_time,
                "gpu_used": gpu_used,
                "model_info": settings.chatbot_model_name,
                "conversation_id": conversation_id
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    async def _prepare_input(self, message: str, conversation_id: Optional[str] = None) -> str:
        """Prepare input text with conversation context from database"""
        if conversation_id:
            try:
                # Get conversation context from database
                context = await database_service.get_conversation_context(
                    conversation_id, max_messages=10
                )
                logger.info(f"Retrieved context for conversation {conversation_id}: {len(context)} messages")

                # Build context string for DialoGPT
                if settings.chatbot_model_name.lower().find("dialogpt") != -1:
                    context_text = ""
                    for ctx_msg in context:
                        if ctx_msg['role'] == 'user':
                            context_text += f"{ctx_msg['content']}{self.tokenizer.eos_token}"
                        else:
                            context_text += f"{ctx_msg['content']}{self.tokenizer.eos_token}"

                    # Add current message
                    final_input = f"{context_text}{message}{self.tokenizer.eos_token}"
                    logger.info(f"Prepared input with context (length: {len(final_input)}): {final_input[:200]}...")
                    return final_input
                else:
                    # For other models, create a conversation format
                    context_text = ""
                    for ctx_msg in context:
                        role_prefix = "User: " if ctx_msg['role'] == 'user' else "Assistant: "
                        context_text += f"{role_prefix}{ctx_msg['content']}\n"

                    return f"{context_text}User: {message}\nAssistant:"

            except Exception as e:
                logger.warning(f"Failed to get conversation context: {e}")
                # Fall back to simple message

        # No conversation context or error - use simple format
        if settings.chatbot_model_name.lower().find("dialogpt") != -1:
            return f"{message}{self.tokenizer.eos_token}"
        else:
            return f"User: {message}\nAssistant:"

    def _clean_response(self, response: str) -> str:
        """Clean and post-process the generated response"""
        # Remove common artifacts
        response = response.replace(self.tokenizer.eos_token, "")
        response = response.replace(self.tokenizer.pad_token, "")

        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and line not in cleaned_lines:
                cleaned_lines.append(line)

        response = '\n'.join(cleaned_lines).strip()

        # Limit response length
        if len(response) > 1000:
            response = response[:1000] + "..."

        return response or "I'm sorry, I couldn't generate a proper response."

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "gpu_memory_used": 0.0,
            "model_info": {
                "name": settings.chatbot_model_name,
                "device": str(self.device),
                "quantized": False,
                "parameters": 0
            }
        }

        # Get GPU information if available
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info["gpu_memory_used"] = gpu.memoryUsed
                    info["gpu_memory_total"] = gpu.memoryTotal
                    info["gpu_utilization"] = gpu.load * 100
                    info["gpu_temperature"] = gpu.temperature
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")

        # Get model information if loaded
        if self.model_loaded and self.model:
            try:
                info["model_info"]["parameters"] = sum(
                    p.numel() for p in self.model.parameters()
                )
                info["model_info"]["quantized"] = hasattr(self.model, 'quantization_config')
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")

        return info

    async def warmup(self):
        """Warmup the model with a test inference"""
        if not self.model_loaded:
            return False

        try:
            logger.info("Warming up model...")
            await self.generate_response("Hello, how are you?")
            logger.info("Model warmup completed")
            return True
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_loaded = False
        logger.info("Chatbot service cleaned up")


# Global chatbot service instance
chatbot_service = OptimizedChatbotService()
