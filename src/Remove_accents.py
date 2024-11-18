import unicodedata
import re

def remove_accents(input_str):
    """
    Remove accents from a string.

    Args:
        input_str (str): The input string with potential accents.

    Returns:
        str: The string with accents removed.
    """
    # Normalize the input string to decompose characters with accents
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Filter out the combining diacritical marks
    accent_removed = ''.join(
        char for char in normalized_str if not unicodedata.combining(char)
    )
    return accent_removed


def divide_string_into_words(dictionary, string):
    # Check if the string is directly in the dictionary
    if string in dictionary:
        return [string]

    word_set = set(dictionary)
    memo = {}

    def can_break(s, last_word_length=None):
        if (s, last_word_length) in memo:
            return memo[(s, last_word_length)]
        if not s:
            # Empty string, return empty list
            return []

        best_result = None
        for i in range(len(s), 0, -1):  # Try longer prefixes first
            prefix = s[:i]
            if prefix in word_set:
                current_word_length = len(prefix)
                if last_word_length == 1 and current_word_length == 1:
                    # Skip if last word and current word are both length 1
                    continue
                suffix = s[i:]
                suffix_result = can_break(suffix, current_word_length)
                if suffix_result is not None:
                    candidate = [prefix] + suffix_result
                    if best_result is None:
                        best_result = candidate
                    else:
                        # Choose the candidate that maximizes the minimal word length
                        min_length_candidate = min(len(word) for word in candidate)
                        min_length_best = min(len(word) for word in best_result)
                        if min_length_candidate > min_length_best:
                            best_result = candidate
                        elif min_length_candidate == min_length_best:
                            if len(candidate) < len(best_result):
                                best_result = candidate
        memo[(s, last_word_length)] = best_result
        return best_result

    result = can_break(string)
    return result if result is not None else [string]

def remove_special_characters(input_string):
    """
    Removes special characters from the input string.
    
    Args:
        input_string (str): The string to process.
    
    Returns:
        str: The processed string with special characters removed.
    """
    return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)