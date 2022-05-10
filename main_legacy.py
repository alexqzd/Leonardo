#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler, ConversationHandler

# Commands
import sys
sys.path.insert(0, './commands')
import printer_control


####
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=11, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]
    
    def get_probability(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=11, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        self.probability_result = dict()
        for idx, prediction in enumerate(self.pred[0]):
            self.probability_result[self.classes[idx]] = prediction
        return self.probability_result


import pickle

from tensorflow.python.keras.models import load_model
model = load_model('models/intents.h5')

with open('utils/classes.pkl','rb') as file:
  classes = pickle.load(file)

with open('utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

with open('utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)


nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
###

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


human_readable_labels = {
    "3d_printer_on": "Encender la impresora 3D",
    "3d_printer_off": "Apagar la impresora 3D",
    "3d_printer_pause": "Pausar la impresión 3D",
    "3d_printer_continue": "Reanudar la impresión 3D",
    "3d_printer_stop": "Cancelar la impresión 3D"
    }

# Convo states
Waiting_for_command, Waiting_for_confirmation = range(2)

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
    if not authorized(user): return
    update.message.reply_markdown_v2(fr'¡Hola {user.mention_markdown_v2()}\! ¿Qué puedo hacer por ti\?')
    return Waiting_for_command

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    if not authorized(update.effective_user): return
    update.message.reply_text('Puedes pedirme que haga cosas, ¡inténtalo!')


def process_message(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    if not authorized(update.effective_user): return
    print(f"Message: {update.message.text.lower()}")
    detected_intents_prob = nlu.get_probability(update.message.text.lower())
    top_intents_buttons = []
    print("Possible intents: ")
    for intent in sorted(detected_intents_prob, key=detected_intents_prob.get, reverse=True):
        if len(top_intents_buttons) >= 3:
            break
        print(intent, f"{detected_intents_prob[intent]*100:.2f}%")
        top_intents_buttons.append([InlineKeyboardButton(human_readable_labels[intent], callback_data=intent)])

    top_intents_buttons.append([InlineKeyboardButton("Cancelar", callback_data="cancel_command")])
    reply_markup = InlineKeyboardMarkup(top_intents_buttons)
    update.message.reply_text('Quieres...', reply_markup=reply_markup)

def confirm(update: Update, context: CallbackContext, function_to_call, confirmation_text, user_answered_yes_message):
    """Asks the user to confirm an action."""
    reply_keyboard = [['Si'],['No']]
    query = update.callback_query
    query.delete_message()
    context.bot.send_message(chat_id=update.effective_chat.id, text=confirmation_text,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder='¿Si o no?'
        ),
    )
    context.user_data['function_to_call'] = function_to_call
    context.user_data['user_answered_yes_message'] = user_answered_yes_message
    return Waiting_for_confirmation

def user_confirmed(update: Update, context: CallbackContext):
    """Handle the user's response."""
    function_to_call = context.user_data['function_to_call']
    
    if update.message.text == "Si":
        try:
            function_to_call()
            update.message.reply_text(context.user_data['user_answered_yes_message'], reply_markup=ReplyKeyboardRemove())
        except Exception as e:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"Me salió este error 😔")
            context.bot.send_message(chat_id=update.effective_chat.id, text=f"{e}")
        del context.user_data['function_to_call']
        return Waiting_for_command
    else:
        update.message.reply_text('Oki entonces no', reply_markup=ReplyKeyboardRemove())
        del context.user_data['function_to_call']
        return Waiting_for_command

def handle_button_press(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    if not authorized(update.effective_user): return
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()
    if query.data == "cancel_command":
        query.edit_message_text(text="Comando cancelado")
        return
    try:
        if query.data == "3d_printer_on":
            printer_control.turn_PSU_on()
            query.edit_message_text(text="Impresora 3D encendida")
            return Waiting_for_command
        elif query.data == "3d_printer_off":
            return confirm(update,context, printer_control.turn_PSU_off, "¿Seguro que quieres apagar la impresora 3D?", "Impresora 3D apagada")
            
        else:
            query.edit_message_text(text="Sorry, aún no se como hacer eso 😬")
            return Waiting_for_command
    except Exception as e:
        query.edit_message_text(text=f"Me salió este error 😔")
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"{e}")
        return Waiting_for_command


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1985053591:AAGBmflL5la-8Nr7e7B_tUnxVzr4IBYwvSg")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            Waiting_for_command: [
                MessageHandler(Filters.text & ~Filters.command, process_message),
                CallbackQueryHandler(handle_button_press)
            ],
            Waiting_for_confirmation: [
                MessageHandler(Filters.regex('^(Si|No)$'), user_confirmed)
            ],
        },
        fallbacks=[CommandHandler('help', help)],
    )

    dispatcher.add_handler(conv_handler)
    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()