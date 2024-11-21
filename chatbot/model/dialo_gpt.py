from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DialoGPTChatbot:
    """Handles interaction with the DialoGPT model."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initializes the DialoGPT model and tokenizer."""
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.chat_history_ids = None

    def generate_response(self, input_text: str) -> str:
        """Generates a response based on the input text."""
        # Encode the input
        new_user_input_ids = self.tokenizer.encode(
            input_text + self.tokenizer.eos_token,
            return_tensors="pt"
        )
        
        # Créer une attention explicite pour les tokens d'entrée
        attention_mask = torch.ones(new_user_input_ids.shape, device=new_user_input_ids.device)

        # Gérer l'historique de conversation
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
            attention_mask = torch.cat([
                torch.ones(self.chat_history_ids.shape, device=self.chat_history_ids.device),
                attention_mask
            ], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Générer une réponse
        with torch.no_grad():
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                attention_mask=attention_mask,  # Utiliser l'attention mask
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                top_p=0.92,
                temperature=0.7,
                do_sample=True,
                top_k=50
            )

        # Décoder la réponse
        bot_response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return bot_response


