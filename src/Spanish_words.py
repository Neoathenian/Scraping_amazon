import nltk
from nltk.corpus import cess_esp
from pattern.es import conjugate, pluralize, singularize
import inflect

from .Remove_accents import remove_special_characters,remove_accents

from wordfreq import word_frequency
import math
from wordfreq import top_n_list

# Get top 1000 Spanish words
spanish_words = top_n_list('es', 1000000)

# Define the minimum frequency threshold (0.01% in decimal form)
threshold = 10**-6

# Filter words by frequency above threshold
filtered_words = []
for word in spanish_words:
    # Get the frequency of the word in the corpus
    frequency = word_frequency(word, 'es')
    
    # Check if the word frequency is above the threshold
    if frequency > threshold:
        filtered_words.append(word)

# Check the number of filtered words
filtered_words = set(remove_accents(word).lower() for word in filtered_words)
filtered_words=set(x for x in filtered_words if len(x)>1 or x in ["o","y","a","e","u"])

spanish_words=filtered_words
#nltk.download('cess_esp')  # Spanish corpus
#
## Get all unique words
#spanish_words = set(word.lower() for word in cess_esp.words())
#spanish_words=set(x for x in spanish_words if len(x)>1 or x in ["o","y","a","e","u"])
#
#
#
#
#inflector = inflect.engine()
#
#def generate_inflections(word):
#    inflections = set()
#
#    # Add the original word
#    inflections.add(word)
#
#    # Pluralize and singularize
#    inflections.add(pluralize(word))
#    inflections.add(singularize(word))
#
#    # Conjugate verbs (if applicable)
#    # We try to conjugate the word in various tenses and persons
#    try:
#        if word.endswith("ar") or word.endswith("er") or word.endswith("ir"):  # Verb check
#            conjugated_forms = conjugate(word)
#            for tense in conjugated_forms:
#                for person in conjugated_forms[tense]:
#                    inflections.add(conjugated_forms[tense][person])
#    except Exception as e:
#        pass  # Skip if conjugation fails (non-verbs)
#
#    # Return all unique inflections for the word
#    return inflections
#
## Process all words and gather all inflected forms
#all_inflections = set()
#for word in spanish_words:
#    word=remove_special_characters(word)
#    all_inflections.update(generate_inflections(word))
#
#spanish_words=all_inflections
#spanish_words=set(x for x in spanish_words if len(x)>1 or x in ["o","y","a","e","u"])