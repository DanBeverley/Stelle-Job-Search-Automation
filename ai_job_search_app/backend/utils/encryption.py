import base64
import logging
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Use centralized configuration for encryption key
ENCRYPTION_KEY = settings.encryption_key
if not ENCRYPTION_KEY:
    raise ValueError("No ENCRYPTION_KEY set in configuration")

# The key must be 16, 24, or 32 bytes long (for AES-128, AES-192, or AES-256).
key = bytes.fromhex(ENCRYPTION_KEY)

def encrypt_data(plain_text_data: str) -> str:
    """
    Encrypts data using AES-256 in GCM mode.
    """
    cipher = AES.new(key, AES.MODE_GCM)
    cipher_text, tag = cipher.encrypt_and_digest(plain_text_data.encode('utf-8'))
    
    encrypted_data = base64.b64encode(cipher.nonce + tag + cipher_text).decode('utf-8')
    return encrypted_data

def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypts data encrypted with AES-256 in GCM mode.
    """
    try:
        decoded_data = base64.b64decode(encrypted_data.encode('utf-8'))
        nonce = decoded_data[:16]
        tag = decoded_data[16:32]
        cipher_text = decoded_data[32:]
        
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plain_text = cipher.decrypt_and_verify(cipher_text, tag).decode('utf-8')
        
        return plain_text
    except (ValueError, KeyError) as e:
        # Handle potential errors during decryption (e.g., tampered data)
        logger.error("Decryption failed: %s", str(e))
        return None 