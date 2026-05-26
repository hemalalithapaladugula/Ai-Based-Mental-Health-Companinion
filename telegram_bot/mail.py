import os
import asyncio
import logging
from fastapi import FastAPI
import httpx
import motor.motor_asyncio
from datetime import datetime
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from twilio.rest import Client
from elevenlabs import ElevenLabs

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MentalHealthBot')

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
ATLAS_URI = os.getenv("ATLAS_URI")
LOCAL_MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER", "")
EMERGENCY_CONTACT = os.getenv("EMERGENCY_CONTACT", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "")
ELEVENLABS_NUMBER_ID = os.getenv("ELEVENLABS_NUMBER_ID", "")

app = FastAPI()
current_date = datetime.now().strftime("%Y-%m-%d")
logger.info("FastAPI application initialized")

class ModelManager:
    def __init__(self, provider: str = "gemini"):
        self.provider = provider.lower()
        logger.info(f"Initializing ModelManager with provider: {self.provider}")
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseChatModel:
        logger.debug(f"Selecting model for provider: {self.provider}")
        try:
            if self.provider == "openai":
                model = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)
                logger.info("OpenAI model initialized")
                return model
            elif self.provider == "gemini":
                model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.2)
                logger.info("Gemini model initialized")
                return model
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    async def run_prompt(self, prompt: str) -> str:
        logger.info(f"Running prompt with {self.provider} model")
        try:
            chain = ChatPromptTemplate.from_template("{prompt}") | self.model | StrOutputParser()
            result = await chain.ainvoke({"prompt": prompt})
            logger.debug(f"Prompt execution successful, result: {result[:50]}...")
            return result.strip()
        except Exception as e:
            logger.error(f"[LLM-{self.provider}] Prompt execution failed: {e}")
            raise

class MentalHealthBot:
    def __init__(self, bot_token: str, mongodb_uri: str):
        logger.info("Initializing MentalHealthBot")
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.client = httpx.AsyncClient(base_url=self.base_url)
        logger.debug("HTTP client initialized for Telegram API")
        self.llm_manager = ModelManager(provider=LLM_PROVIDER)
        self.mongodb_uri = mongodb_uri
        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_uri)
        self.db = self.mongo_client["mental_health_bot"]
        self.conversations = self.db["conversations"]
        self.summaries = self.db["summaries"]
        self.last_update_id = 0
        self.is_mongo_connected = True
        self.twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None
        logger.info(f"Twilio client {'initialized' if self.twilio_client else 'not configured'}")
        self.elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
        self.elevenlabs_agent_id = ELEVENLABS_AGENT_ID
        self.elevenlabs_number_id = ELEVENLABS_NUMBER_ID
        self.emergency_contact = EMERGENCY_CONTACT
        logger.info(f"ElevenLabs client {'initialized' if self.elevenlabs_client else 'not configured'}")

        # Attempt MongoDB connection
        try:
            atlas_client = MongoClient(self.mongodb_uri, server_api=ServerApi('1'))
            atlas_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}. Continuing without database.")
            self.is_mongo_connected = False

    async def get_updates(self):
        logger.debug(f"Fetching Telegram updates with offset: {self.last_update_id + 1}")
        try:
            r = await self.client.get(f"/getUpdates?offset={self.last_update_id + 1}")
            logger.info("Successfully fetched Telegram updates")
            return r.json()
        except Exception as e:
            logger.error(f"Error fetching Telegram updates: {e}")
            return {"result": []}

    async def send_message(self, chat_id: int, text: str):
        logger.info(f"Sending message to chat_id {chat_id}: {text[:50]}...")
        try:
            await self.client.post("/sendMessage", json={"chat_id": chat_id, "text": text})
            logger.debug("Message sent successfully")
        except Exception as e:
            logger.error(f"Error sending message to chat_id {chat_id}: {e}")

    async def send_chat_action(self, chat_id: int, action: str):
        logger.debug(f"Sending chat action {action} to chat_id {chat_id}")
        try:
            await self.client.post("/sendChatAction", json={"chat_id": chat_id, "action": action})
            logger.debug("Chat action sent successfully")
        except Exception as e:
            logger.error(f"Error sending chat action to chat_id {chat_id}: {e}")

    async def get_user_summary(self, user_id: int) -> str:
        logger.info(f"Retrieving summary for user_id: {user_id}")
        if not self.is_mongo_connected:
            logger.warning("MongoDB not connected, skipping summary retrieval")
            return "No past conversations available due to a database issue."
        try:
            doc = await self.summaries.find_one({"user_id": user_id}, sort=[("timestamp", -1)])
            if doc and "summary" in doc:
                logger.debug(f"Summary found for user_id {user_id}")
                return doc["summary"]
            logger.info(f"No summary found for user_id {user_id}")
            return "No past conversations found."
        except Exception as e:
            logger.error(f"Failed to retrieve user summary for user_id {user_id}: {e}")
            return "No past conversations available due to a database issue."

    async def update_user_summary(self, user_id: int, new_msg: str, response: str, prev_summary: str):
        logger.info(f"Updating summary for user_id: {user_id}")
        if not self.is_mongo_connected:
            logger.warning("MongoDB not connected, skipping summary update")
            return
        prompt = (
            f"Previous summary: {prev_summary}\n"
            f"New user message: {new_msg}\n"
            f"AI response: {response}\n"
            "Generate a concise updated summary of the user's emotional well-being and conversation history so far, focusing on key themes, mood trends, and progress."
        )
        try:
            new_sum = await self.llm_manager.run_prompt(prompt)
            await self.summaries.update_one(
                {"user_id": user_id},
                {"$set": {"summary": new_sum, "timestamp": datetime.now()}},
                upsert=True
            )
            logger.debug(f"Summary updated successfully for user_id {user_id}")
        except Exception as e:
            logger.error(f"Summary update failed for user_id {user_id}: {e}")

    async def store_conversation(self, user_id: int, username: str, msg: str, response: str, report: bool, escalate: bool):
        logger.info(f"Storing conversation for user_id: {user_id}")
        if not self.is_mongo_connected:
            logger.warning("MongoDB not connected, skipping conversation storage")
            return
        try:
            await self.conversations.insert_one({
                "user_id": user_id,
                "username": username,
                "message": msg,
                "response": response,
                "report": report,
                "escalate": escalate,
                "timestamp": datetime.now()
            })
            logger.debug(f"Conversation stored successfully for user_id {user_id}")
        except Exception as e:
            logger.error(f"Failed to store conversation for user_id {user_id}: {e}")

    async def check_escalation(self, msg: str) -> bool:
        logger.debug("Checking for escalation keywords")
        keywords = ["suicide", "kill myself", "end my life", "harm myself", "want to die"]
        result = any(kw.lower() in msg.lower() for kw in keywords)
        logger.info(f"Escalation check: {'Triggered' if result else 'Not triggered'}")
        return result

    async def check_for_call_request(self, msg: str) -> bool:
        logger.debug("Checking for call request")
        result = "personal call" in msg.lower() or "1-1 call" in msg.lower()
        logger.info(f"Call request check: {'Triggered' if result else 'Not triggered'}")
        return result

    def trigger_escalation(self, user_id: int, msg: str):
        logger.info(f"Triggering escalation for user_id {user_id}")
        if not self.twilio_client:
            logger.warning("Twilio not configured, skipping escalation")
            return
        message = f"Alert: User {user_id} may be in distress. Their message: {msg}. Please check on them."
        try:
            self.twilio_client.calls.create(
                to=self.emergency_contact,
                from_=TWILIO_FROM,
                twiml=f'<Response><Say voice="woman">{message}</Say></Response>'
            )
            logger.info(f"Escalation call triggered successfully for user_id {user_id}")
        except Exception as e:
            logger.error(f"Twilio call failed for user_id {user_id}: {e}")

    def trigger_elevenlabs_call(self, to_number: str, username: str = "", context: str = ""):
        logger.info(f"Triggering ElevenLabs call to {to_number}")
        if not self.elevenlabs_client or not self.elevenlabs_agent_id or not self.elevenlabs_number_id:
            logger.warning("ElevenLabs not fully configured, skipping call")
            return
        try:
            call_data = self.elevenlabs_client.conversational_ai.twilio.outbound_call(
                agent_id=self.elevenlabs_agent_id,
                agent_phone_number_id=self.elevenlabs_number_id,
                to_number=to_number,
                conversation_initiation_client_data={
                    "dynamic_variables": {
                        "user_name": username or "User",
                        "context": context or "Personal 1-1 mental health support call."
                    }
                }
            )
            logger.info(f"ElevenLabs call triggered successfully: {call_data}")
        except Exception as e:
            logger.error(f"ElevenLabs call failed: {e}")

    async def generate_response(self, msg: str, username: str, user_id: int, report: bool, escalate: bool) -> str:
        logger.info(f"Generating response for user_id {user_id}, report: {report}, escalate: {escalate}")
        if report:
            logger.debug("Handling report request")
            summary = await self.get_user_summary(user_id)
            prompt = (
                f"Analyze the following conversation summary and provide a detailed emotional progress report over time:\n{summary}"
            )
            try:
                response = await self.llm_manager.run_prompt(prompt)
                logger.debug(f"Report generated successfully for user_id {user_id}")
                return response if response.strip() else "No progress report available at this time."
            except Exception as e:
                logger.error(f"Report generation error for user_id {user_id}: {e}")
                return "Sorry, I couldn't generate your progress report right now."

        if escalate:
            logger.info("Returning escalation response")
            return "I'm here for you. I'll notify a specialist right away."

        if await self.check_for_call_request(msg):
            logger.info(f"Initiating ElevenLabs call for user_id {user_id}")
            self.trigger_elevenlabs_call(str(user_id), username, msg)
            return "Initiating a personal 1-1 call with you shortly."

        summary = await self.get_user_summary(user_id)
        prompt = f"""
        You are a multilingual AI mental health companion designed to support users with empathy, clarity, and psychological guidance.
        Your role:
        - Act as a friendly, supportive, non-judgmental therapist trained in CBT and culturally sensitive care.
        - The conversation should feel like a continuous dialogue, not like a fresh message each time.
        - Always respond in a warm, calm, and professional manner, using simple language.
        - Use simple, compassionate language to help users understand and manage emotional challenges.
        - If the user messages in their language, reply in their language with English words where appropriate.
        - Respond based on both the current message and the user's emotional history (if memory summary is available).
        - Offer gentle nudges toward mental well-being through interactive suggestions like mood-check-ins, self-reflection questions, breathing exercises, games, or helpful resources.
        - If the user appears in distress or mentions suicidal thoughts, offer comforting words and subtly suggest talking to a human therapist. (Escalation is handled separately.)
        - Username: {username}
        User's conversation summary: {summary}
        Current date: {current_date}
        Respond to the following message with this mindset: {msg}
        """
        try:
            response = await self.llm_manager.run_prompt(prompt)
            logger.debug(f"Response generated successfully for user_id {user_id}: {response[:50]}...")
            return response if response.strip() else "I'm here to help, but I'm having trouble responding right now. Please try again later."
        except Exception as e:
            logger.error(f"LLM response generation failed for user_id {user_id}: {e}")
            return "Sorry, I'm having trouble responding right now. Please try again later."

    async def process_update(self, update: dict):
        self.last_update_id = update['update_id']
        msg = update['message']['text']
        chat_id = update['message']['chat']['id']
        user = update['message']['from']
        user_id = user.get('id')
        username = user.get('username', 'User')
        logger.info(f"Processing update {self.last_update_id} for user_id {user_id}, message: {msg[:50]}...")

        await self.send_chat_action(chat_id, 'typing')
        report = any(keyword in msg.lower() for keyword in ['report', 'progress'])
        escalate = await self.check_escalation(msg)
        logger.debug(f"Intent detection - Report: {report}, Escalate: {escalate}")

        if escalate:
            logger.info(f"Escalation triggered for user_id {user_id}")
            self.trigger_escalation(user_id, msg)

        response = await self.generate_response(msg, username, user_id, report, escalate)
        await self.send_message(chat_id, response)

        if report and not escalate:
            logger.info(f"Sending report confirmation for user_id {user_id}")
            await self.send_message(chat_id, "Your progress report has been sent above.")

        await self.store_conversation(user_id, username, msg, response, report, escalate)
        if not escalate and not report:
            logger.debug(f"Updating summary for user_id {user_id}")
            prev_summary = await self.get_user_summary(user_id)
            await self.update_user_summary(user_id, msg, response, prev_summary)
        logger.info(f"Update {self.last_update_id} processed successfully for user_id {user_id}")

    async def poll_updates(self):
        logger.info("Starting Telegram update polling")
        while True:
            updates = await self.get_updates()
            logger.debug(f"Received {len(updates.get('result', []))} updates")
            for upd in updates.get('result', []):
                await self.process_update(upd)
            await asyncio.sleep(0.5)

bot = MentalHealthBot(BOT_TOKEN, ATLAS_URI or LOCAL_MONGODB_URI)
logger.info("MentalHealthBot instance created")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI lifespan")
    task = asyncio.create_task(bot.poll_updates())
    yield
    logger.info("Shutting down FastAPI lifespan")
    task.cancel()
    await bot.client.aclose()
    logger.debug("HTTP client closed")
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Polling task cancelled")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"status": "Bot is running"}




# test 333 with eleven labs