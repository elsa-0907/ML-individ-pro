"""
fact_checker.py - World Knowledge & Semantic Fact-Checking Engine
Verifies claims against factual ontologies (botany, astronomy, anatomy, viral hoaxes).
"""

import re
from typing import Optional, Tuple, Dict

# 1. Botanical / Biological taxonomy
FRUITS = {
    "apple", "banana", "orange", "grape", "mango", "strawberry", "blueberry",
    "watermelon", "pineapple", "peach", "pear", "cherry", "lemon", "lime",
    "plum", "papaya", "avocado", "tomato"
}

VEGETABLES = {
    "carrot", "broccoli", "spinach", "potato", "onion", "garlic", "lettuce",
    "cabbage", "cauliflower", "cucumber", "pepper", "celery", "asparagus"
}

ANIMALS_TAXONOMY = {
    "whale": ("mammal", ["fish", "reptile", "bird", "insect"]),
    "dolphin": ("mammal", ["fish", "reptile", "bird", "insect"]),
    "bat": ("mammal", ["bird", "insect", "reptile", "fish"]),
    "spider": ("arachnid", ["insect", "mammal", "reptile", "bird"]),
    "penguin": ("flightless bird", ["mammal", "fish", "reptile"]),
    "shark": ("cartilaginous fish", ["mammal", "reptile", "amphibian"]),
    "frog": ("amphibian", ["reptile", "mammal", "insect", "fish"]),
    "snake": ("reptile", ["mammal", "amphibian", "insect", "bird"])
}

# 2. Known debunked hoaxes & pseudoscientific falsehoods
DEBUNKED_FACTS = [
    (r"\b(earth|world)\s+is\s+flat\b", 
     "Scientific consensus confirms Earth is an oblate spheroid, not flat."),
    (r"\b(sun\s+revolves\s+around\s+(the\s+)?earth)\b", 
     "Heliocentrism proves Earth revolves around the Sun."),
    (r"\b(sun\s+is\s+a\s+planet)\b", 
     "The Sun is an astronomical star, not a planet."),
    (r"\b(moon\s+is\s+made\s+of\s+cheese)\b", 
     "The Moon is composed of silicate rock and metallic elements."),
    (r"\b(vaccines?\s+contain\s+(5g|microchips?|tracking))\b", 
     "Debunked conspiracy theory: vaccines do not contain microchips or 5G transmitters."),
    (r"\b(drinking\s+bleach\s+cures?)\b", 
     "Dangerous health misinformation: consuming bleach is toxic and does not cure illnesses."),
    (r"\b(covid(-19)?\s+is\s+caused\s+by\s+5g)\b", 
     "Scientific consensus shows viruses are biological pathogens, unrelated to radio frequency waves."),
    (r"\b(gravity\s+does(n'?t|\s+not)\s+exist)\b", 
     "Gravity is a fundamental physical force governing mass interactions in physics.")
]


def check_factual_veracity(text: str) -> Optional[Tuple[bool, str]]:
    """
    Check if text contains verifiable factual falsehoods.
    Returns:
        (is_false: bool, explanation: str) if a verifiable factual error is detected,
        otherwise None.
    """
    clean = text.lower().strip()
    clean = re.sub(r"[^\w\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # 1. Check known debunked hoaxes & scientific contradictions
    for pattern, explanation in DEBUNKED_FACTS:
        if re.search(pattern, clean):
            return True, explanation

    # 2. Check Botanical / Food category contradictions
    # Pattern: [fruit] is (a/an) vegetable / meat
    for fruit in FRUITS:
        # e.g., "apple is a vegetable", "apples are vegetables"
        pattern_veg = rf"\b{fruit}s?\s+(is|are)\s+(an?\s+)?vegetables?\b"
        if re.search(pattern_veg, clean):
            return True, f"Botanical fact check: {fruit.capitalize()} is classified as a fruit, not a vegetable."

        pattern_meat = rf"\b{fruit}s?\s+(is|are)\s+(an?\s+)?(meat|poultry|dairy)\b"
        if re.search(pattern_meat, clean):
            return True, f"Botanical fact check: {fruit.capitalize()} is a plant fruit, not {re.search(pattern_meat, clean).group(3)}."

    for veg in VEGETABLES:
        pattern_fruit = rf"\b{veg}s?\s+(is|are)\s+(an?\s+)?fruits?\b"
        if re.search(pattern_fruit, clean):
            return True, f"Botanical fact check: {veg.capitalize()} is a vegetable, not a fruit."

    # 3. Check Animal Taxonomy contradictions
    for animal, (correct_class, wrong_classes) in ANIMALS_TAXONOMY.items():
        for wrong in wrong_classes:
            pattern_animal = rf"\b{animal}s?\s+(is|are)\s+(an?\s+)?{wrong}s?\b"
            if re.search(pattern_animal, clean):
                return True, (
                    f"Biological fact check: {animal.capitalize()}s are classified as {correct_class}s, "
                    f"not {wrong}."
                )

    # 4. Check Anatomy contradictions
    if re.search(r"\bhumans?\s+(have|has)\s+(3|three|4|four|5|five)\s+(eyes?|heads?|hearts?)\b", clean):
        return True, "Anatomy fact check: Humans biologically have two eyes, one head, and one heart."

    if re.search(r"\bwater\s+is\s+(dry|toxic\s+to\s+drink)\b", clean):
        return True, "Physical property fact check: Water (H2O) is a liquid at standard temperature and is essential for life."

    return None
