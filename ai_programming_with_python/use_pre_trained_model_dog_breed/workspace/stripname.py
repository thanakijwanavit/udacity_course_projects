#!/usr/bin/env python3
def get_label_fromfile(filename):
    # Sets pet_image variable to a filename 
    pet_image = filename
    # Sets string to lower case letters
    low_pet_image = pet_image.lower()
    # Splits lower case string by _ to break into words 
    word_list_pet_image = low_pet_image.split("_")
    # Create pet_name starting as empty string
    pet_name = ""
    # Loops to check if word in pet name is only
    # alphabetic characters - if true append word
    # to pet_name separated by trailing space 
    for word in word_list_pet_image:
        if word.isalpha():
            pet_name += word + " "
    # Strip off starting/trailing whitespace characters 
    pet_name = pet_name.strip()
    return pet_name
