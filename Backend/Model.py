import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.simplefilter('ignore')

from openai import OpenAI

class PichuModel:
    def __init__(self, api_key: str):
        self.intents = self._load_intents()
        self.training_data = []
        self.labels = []
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                self.training_data.append(pattern.lower())
                self.labels.append(intent)
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), stop_words="english", max_df=0.8, min_df=1)
        X = self.vectorizer.fit_transform(self.training_data)
        X_train, X_test, Y_train, Y_test = train_test_split(X, self.labels, test_size=0.4, random_state=42, stratify=self.labels)
        self.model = SVC(kernel='linear', probability=True, C=1.0)
        self.model.fit(X_train, Y_train)
        # NVIDIA API client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def _load_intents(self):
        return {
            "greetings": {
                "patterns": ["hello", "hi", "hey", "what's up", "how are you", "good morning", "good afternoon", "good evening", "hola", "yo"],
                "responses": ["Hey there! I'm Pichu, your AI assistant. How can I help you today?", "Hi! I'm Pichu! What can I do for you?", "Hello! I'm Pichu. How can I assist you?"]
            },
            "goodbye": {
                "patterns": ["bye", "see you later", "goodbye", "farewell", "catch you later", "see you soon", "quit", "exit"],
                "responses": ["Goodbye!", "See you later!", "Have a great day!"]
            },
            "gratitude": {
                "patterns": ["thank you", "thanks", "appreciate it", "thank you so much", "thanks a lot", "much appreciated", "thx"],
                "responses": ["You're welcome!", "Happy to help!", "Glad I could assist.", "Anytime! Is there anything else I can help with?"]
            },
            "apologies": {
                "patterns": ["sorry", "my apologies", "apologize", "i'm sorry", "sorry about that"],
                "responses": ["No problem at all.", "It's alright.", "No need to apologize.", "That's okay."]
            },
            "positive_feedback": {
                "patterns": ["great job", "well done", "awesome", "fantastic", "amazing work", "excellent", "brilliant", "wonderful", "good pichu"],
                "responses": ["Thank you! I appreciate your feedback.", "Glad to hear that!", "Thank you for the compliment!"]
            },
            "negative_feedback": {
                "patterns": ["not good", "disappointed", "unsatisfied", "poor service", "needs improvement", "could be better", "not helpful"],
                "responses": ["I'm sorry to hear that. Can you please provide more details so I can assist you better?", "I apologize for the inconvenience. Let me help resolve the issue."]
            },
            "help": {
                "patterns": ["help", "can you help me", "i need assistance", "support", "what can you do", "how do you help"],
                "responses": ["Sure, I'll do my best to assist you.", "Of course, I'm here to help!", "How can I assist you?"]
            },
            "name": {
                "patterns": ["what's your name", "what is your name", "tell me about yourself", "what are you", "identify yourself", "who are you", "who created you", "your name", "who is pichu"],
                "responses": [
                    "My name is Pichu! I'm your personal AI assistant powered by NVIDIA Minmax.",
                    "I am Pichu, your personal AI assistant.",
                    "I'm Pichu! I was created by my Master and I'm powered by NVIDIA Minmax."
                ]
            },
            "jokes": {
                "patterns": ["tell me a joke", "joke please", "got any jokes", "make me laugh", "say something funny", "crack a joke", "jokes"],
                "responses": [
                    "Why did the chicken cross the road? To get to the other side!",
                    "I'm reading a book on anti-gravity. It's impossible to put down!",
                    "Why did the computer show up at work late? It had a hard drive!",
                    "Why do programmers prefer dark mode? Because light attracts bugs!",
                    "Why did the chicken join a band? Because it had the drumsticks!",
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "Why did the scarecrow win an award? Because he was outstanding in his field!",
                    "What do you call a fake noodle? An impasta!"
                ]
            },
            "facts": {
                "patterns": ["tell me a fact", "fact please", "got any fun fact", "you know any facts", "did you know", "interesting fact", "facts"],
                "responses": [
                    "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs!",
                    "Bananas are berries, but strawberries aren't.",
                    "Octopuses have three hearts.",
                    "A group of flamingos is called a 'flamboyance'.",
                    "The shortest war in history lasted 38 minutes.",
                    "Venus is the only planet that spins clockwise.",
                    "Dolphins sleep with one eye open!"
                ]
            },
            "weather_query": {
                "patterns": ["how's the weather", "what's the weather", "is it raining", "will it rain today", "weather forecast"],
                "responses": ["I can check the weather for you! Just ask 'weather in [city]'."]
            },
            "music_query": {
                "patterns": ["play some music", "put on a song", "i want to hear music", "music please"],
                "responses": ["I can play music for you! Just say 'play music' or 'play [song name]'."]
            }
        }

    def predict_intent(self, user_input):
        user_input = user_input.lower()
        try:
            input_vector = self.vectorizer.transform([user_input])
            intent = self.model.predict(input_vector)[0]
            return intent
        except Exception:
            return None

    def get_response(self, intent):
        if intent in self.intents:
            return random.choice(self.intents[intent]['responses'])
        return None

    def handle_chat(self, user_input):
        intent = self.predict_intent(user_input)
        return self.get_response(intent)

    def handle_ai_query(self, query, language="en"):
        try:
            # Add context for Pichu assistant - always identify as Pichu
            enhanced_query = f"You are Pichu, a friendly AI assistant. If someone asks who you are, always say you are Pichu. Answer concisely in 2-3 sentences: {query}"

            completion = self.client.chat.completions.create(
                model="minimaxai/minimax-01",
                messages=[{"role": "user", "content": enhanced_query}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=512,
                stream=False
            )
            response = completion.choices[0].message.content.strip()
            # Ensure the response starts with Pichu identity if asked about name
            query_lower = query.lower()
            if any(x in query_lower for x in ["who are you", "what is your name", "what are you", "your name"]):
                if not response.lower().startswith("pichu"):
                    response = "I'm Pichu, your AI assistant. " + response
            return response
        except Exception as e:
            return "I'm Pichu! I'm having trouble processing that request. Please check the command and try again, or visit the Commands panel for available options."

    def get_intent_response(self, query):
        query_lower = query.lower()
        for intent, data in self.intents.items():
            for pattern in data.get("patterns", []):
                if pattern in query_lower:
                    responses = data.get("responses", [])
                    if responses:
                        return random.choice(responses)
        return None  # No match found
