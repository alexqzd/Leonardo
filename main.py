#!/usr/bin/env python
# pylint: disable=C0116,W0613

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler, ConversationHandler
import openai
import os
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

telegram_token = os.environ['TELEGRAM_TOKEN']
openai_api_key = os.environ['OPENAI_API_KEY']
if 'WEBHOOK_URL' in os.environ:
    webhook_url = os.environ['WEBHOOK_URL']

# Enable logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

openai.api_key = openai_api_key

gpt_3_engines = ["text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-002"]
current_engine = gpt_3_engines[3] 

AI_name = "AI"
_chat_log = []

def append_to_chat_log(username: str, message: str):
    """Append a message to the chat log."""
    _chat_log.append(f"{username}: {message.strip()}")

def get_chat_log() -> str:
    """Get the chat log."""
    return "\n".join(_chat_log)

def get_chat_response(username: str, message: str) -> str:
    """Get the chat response."""
    append_to_chat_log(username, message)
    print(f"Current chatlog is:\n{get_chat_log()}")
    response = openai.Completion.create(
        engine=current_engine,
        prompt = f"{get_chat_log()}\n{username}: {message}\n{AI_name}:",
        stop = [f"{username}:", f"{AI_name}:"],
        temperature=0.9,
        max_tokens=200,
        top_p=1,
        frequency_penalty=1.3,
        presence_penalty=1
    )
    response = response.choices[0].text
    append_to_chat_log(AI_name, response)
    return response

# Convo states
Waiting_for_chat_message, Waiting_for_raw_prompt = range(2)

whitelist = [1080659616]

def authorized(user) -> bool:
    """Check if the user is authorized."""
    if user.id not in whitelist:
        print(f"User {user.full_name} ({user.id}) is not authorized")
        return False
    else:
        return True

# Define a few command handlers. These usually take the two arguments update and
# context.

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    if not authorized(update.effective_user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    print(f"Message: {update.message.text.lower()}")
    response = openai.Completion.create(
        engine=current_engine,
        prompt=f"You are a chatbot named Leonardo. Write a message greeting {update.effective_user.full_name}.",
        temperature=0.8,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )
    response = response.choices[0].text
    update.message.reply_text(response)
    _chat_log.clear()
    append_to_chat_log(AI_name, response)
    return Waiting_for_chat_message

def process_message(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    if not authorized(update.effective_user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    message = update.message.text
    print(f"Message: {message}")
    response = get_chat_response(update.effective_user.first_name, message)
    update.message.reply_text(response)
    print(f"Response: {response}")
    return Waiting_for_chat_message

def process_voice_message(update: Update, context: CallbackContext) -> None:
    if not authorized(update.effective_user): return
    # get basic info about the voice note file and prepare it for downloading
    new_file = context.bot.get_file(update.message.voice.file_id)
    # download the voice note as a file
    new_file.download("voice_note.ogg")
    sound = AudioSegment.from_ogg("voice_note.ogg")
    sound.export("voice_note.wav", format="wav")
    r = sr.Recognizer()
    with sr.AudioFile("voice_note.wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        message = r.recognize_google(audio_data)
        context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        print(f"Voice message: {message}")
        response = get_chat_response(update.effective_user.first_name, f"[Voice message transcription]: {message}")
        tts = gTTS(text=response, lang="en", slow=False)
        tts.save("reply.mp3")
        # sound = AudioSegment.from_wav("reply.wav")
        #sound.export("reply.mp3", format="mp3")
        update.message.reply_voice(voice=open("reply.mp3", "rb"))
        print(f"Response: {response}")
        return Waiting_for_chat_message

def process_raw_prompt(update: Update, context: CallbackContext) -> None:
    """Pass the raw prompt to the AI."""
    if not authorized(update.effective_user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    message = update.message.text
    print(f"Message: {message}")
    response = openai.Completion.create(
        engine=current_engine,
        prompt=message,
        temperature=0.9,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )
    response = response.choices[0].text
    print(f"Raw response: {response}")
    update.message.reply_text(response)
    return Waiting_for_raw_prompt

def switch_to_raw_prompt(update: Update, context: CallbackContext) -> None:
    """Switch to raw prompt."""
    if not authorized(update.effective_user): return
    update.message.reply_text("I will now pass your messages directly to OpenAI. Write a message to continue.")
    return Waiting_for_raw_prompt

def switch_to_chat_mode(update: Update, context: CallbackContext) -> None:
    """Switch to chat mode.""" 
    if not authorized(update.effective_user): return
    update.message.reply_text("Chat mode is now enabled, I try to keep track of our recent conversation.")
    update.message.reply_text("If something goes wrong, you can restart the conversation by typing /start.")
    update.message.reply_text("Write a message to continue.")
    return Waiting_for_chat_message

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    if not authorized(update.effective_user): return
    update.message.reply_text('You can ask me things, try it!')
    update.message.reply_text('You can switch to raw prompt mode by typing /raw_mode')
    update.message.reply_text('You can switch to chat mode by typing /chat_mode (this is the default)')
    update.message.reply_text('You can reset the chat mode conversation by typing /start')

def process_start_message(update: Update, context: CallbackContext) -> None:
    """Reply the user message."""
    print("The user has started the conversation without the /start command.")
    if not authorized(update.effective_user): return
    update.message.reply_text('Hi! Send the /start command to start the conversation.')

def get_available_engines(update: Update, context: CallbackContext) -> None:
    """Get the available engines and show them to the user."""
    if not authorized(update.effective_user): return
    print("The user has requested the available engines.")
    available_engines_buttons = []
    print("Available engines:")
    for engine in gpt_3_engines:
        print(engine)
        available_engines_buttons.append([InlineKeyboardButton(engine, callback_data=engine)])

    available_engines_buttons.append([InlineKeyboardButton("Cancel", callback_data="cancel_command")])
    reply_markup = InlineKeyboardMarkup(available_engines_buttons)
    update.message.reply_text(f"Currently using {current_engine}.\nChoose another engine to use:", reply_markup=reply_markup)
    
def handle_button_press(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    if not authorized(update.effective_user): return
    query = update.callback_query
    global current_engine
    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()
    if query.data == "cancel_command":
        query.edit_message_text(text=f"OK, I'll keep using {current_engine}.")
        return
    else:
        current_engine = query.data
        query.edit_message_text(text=f"OK, I'll use {current_engine} from now on.")

mode_switch_commands_handler = [CommandHandler('chat_mode', switch_to_chat_mode), CommandHandler('raw_mode', switch_to_raw_prompt), CommandHandler('switch_engine', get_available_engines), CallbackQueryHandler(handle_button_press)]


def main() -> None:
    """Start the bot."""
    print("Starting bot...")
    
    # Create the Updater and pass it your bot's token.
    updater = Updater(telegram_token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.text & ~Filters.command, process_start_message), CommandHandler('start', start)],
        states={
            Waiting_for_chat_message: [
                MessageHandler(Filters.text & ~Filters.command, process_message),
                MessageHandler(Filters.voice , process_voice_message),
                CommandHandler('help', help_command),CommandHandler('start', start)] + mode_switch_commands_handler,
            Waiting_for_raw_prompt: [
                MessageHandler(Filters.text & ~Filters.command,process_raw_prompt),
                CommandHandler('help', help_command)] + mode_switch_commands_handler,
        },
        fallbacks=[CommandHandler('help', help)],
    )

    dispatcher.add_handler(conv_handler)

    PORT = int(os.environ.get('PORT', '8443'))

    # Start the Bot
    if 'WEBHOOK_URL' in os.environ:
        updater.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=telegram_token,
            webhook_url=webhook_url + telegram_token
        )
        print("Started webhook")
    else:
        updater.start_polling()
        print("Started polling")
        
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()
