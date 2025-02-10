import json

all_targets = {
    'Cassette Player': ['Typewriter'],
    'Chain Saw': ['Chain'],
    'Church': ['Temple'],
    'Gas Pump': ['ATM'],
    'Tench': ['Fish'],
    'Garbage Truck': ['Bus'],
    'English Springer': ['Dog'],
    'Golf Ball': ['Tennis Ball'],
    'parachute': ['umbrella'],
    'French Horn': ['trombone'],
}

# ['English Springer', 'Clumber Spaniel', 'English Setter', 'Blenheim Spaniel', 'Border Collie',
# 'Garbage Truck', 'Moving Van', 'Fire Engine', 'Ambulance', 'School Bus',
# 'French Horn', 'Bassoon', 'Trombone', 'Oboe', 'Saxophone',
# 'Church', 'Monastery', 'Bell Cote', 'Dome', 'Library',
# 'Cassette Player', 'Polaroid Camera', 'Loudspeaker', 'Typewriter Keyboard', 'Projector']

all_targets_v2 = {
    'Cassette Player': ['Polaroid Camera'],
    'Chain Saw': ['Chain'],
    'Church': ['Monastery'],
    'Gas Pump': ['Petrol Pump'],
    'Tench': ['Fish'],
    'Garbage Truck': ['Moving Van'],
    'English Springer': ['Clumber Spaniel'],
    'Golf Ball': ['Tennis Ball'],
    'parachute': ['umbrella'],
    'French Horn': ['Trombone'],
}

all_targets_netfive = {
    'English Springer': ['dog'],
    'Clumber Spaniel': ['dog'],
    'English Setter': ['dog'],
    'Blenheim Spaniel': ['dog'],
    'Border Collie': ['dog'],
    'Garbage Truck': ['car'],
    'Moving Van': ['car'],
    'Fire Engine': ['car'],
    'Ambulance': ['car'],
    'School Bus': ['car'],
    'French Horn': ['instrument'],
    'Bassoon': ['instrument'],
    'Trombone': ['instrument'],
    'Oboe': ['instrument'],
    'Saxophone': ['instrument'],
    'Church': ['building'],
    'Monastery': ['building'],
    'Bell Cote': ['building'],
    'Dome': ['building'],
    'Library': ['building'],
    'Cassette Player': ['camera'],
    'Polaroid Camera': ['camera'],
    'Loudspeaker': ['camera'],
    'Typewriter Keyboard': ['camera'],
    'Projector': ['camera'],
}

def get_target(concept):
    if concept in all_targets_netfive:
        return all_targets_netfive[concept][0]
    else:
        return None


def get_prompt(prompt):
    preserved = ' '

    if prompt == 'all_artists':
        with open('./data/MACE_erase_art_100.json', 'r') as f:
            all_artists = json.load(f)
        prompt = ', '.join(all_artists['erase'])
        preserved = ' '

    if prompt == 'Kelly McKernan':
        prompt = "Kelly Mckernan"
        preserved = ' '
    
    if prompt == 'Thomas Kinkade':
        prompt = "Thomas Kinkade"
        preserved = ' '
    
    if prompt == 'Ajin Demi Human':
        prompt = "Ajin Demi Human"
        preserved = ' '
        
    if prompt == 'Tyler Edlin':
        prompt = "Tyler Edlin"
        preserved = ' '
    
    if prompt == 'Kilian Eng':
        prompt = "Kilian Eng"
        preserved = ' '

    if prompt == 'Van Gogh':
        prompt = "Van Gogh"
        preserved = ' '

    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
        preserved = ' '

    if prompt == 'nudity':
        prompt = "nudity"
        preserved = ' '

    if prompt == 'nudity_with_person':
        prompt = "nudity"
        preserved = 'person'

    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
    
    if prompt == 'imagenette':
        prompt = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'parachute', 'French Horn']
        prompt = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench, Garbage Truck, English Springer, Golf Ball, parachute, French Horn'
    
    if prompt == 'imagenette_small':
        prompt = 'Cassette Player, Church, Garbage Truck, parachute, French Horn'
        preserved = 'Chain Saw, Gas Pump, Tench, English Springer, Golf Ball'

    if prompt == 'imagenette_v2':
        prompt = 'Chain Saw, Gas Pump, Tench, English Springer, Golf Ball'
        preserved = 'Cassette Player, Church, Garbage Truck, parachute, French Horn'
    
    if prompt == 'imagenette_v3':
        prompt = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench'
        preserved = 'Garbage Truck, English Springer, Golf Ball, parachute, French Horn'
    
    if prompt == 'imagenette_v4':
        prompt = 'Garbage Truck, English Springer, Golf Ball, parachute, French Horn'
        preserved = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench'
    
    if prompt == 'cassette_player':
        prompt = 'Cassette Player'
        preserved = ' '

    if prompt == 'garbage_truck':
        prompt = 'Garbage Truck'
        preserved = ' '

    if prompt == 'garbage_truck_with_lexus':
        prompt = 'Garbage Truck'
        preserved = 'lexus'

    if prompt == 'garbage_truck_with_road':
        prompt = 'Garbage Truck'
        preserved = 'road'

    if prompt == 'imagenette_v1_wo':
        prompt = 'Cassette Player, Church, Garbage Truck, parachute, French Horn'
        preserved = ' '

    if prompt == 'imagenette_v2_wo':
        prompt = 'Chain Saw, Gas Pump, Tench, English Springer, Golf Ball'
        preserved = ' '
    
    if prompt == 'imagenette_v3_wo':
        prompt = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench'
        preserved = ' '
    
    if prompt == 'imagenette_v4_wo':
        prompt = 'Garbage Truck, English Springer, Golf Ball, parachute, French Horn'
        preserved = ' '

    if prompt == 'taylor swift':
        prompt = 'Taylor Swift'
        preserved = ' '
    
    if prompt == 'gun':
        prompt = 'gun'
        preserved = ' '
    
    if prompt == 'margot robbie':
        prompt = 'Margot Robbie'
        preserved = ' '
    
    if prompt == 'barack obama':
        prompt = 'Barack Obama'
        preserved = ' '
        
    if prompt == 'netfive':
        prompt = 'English Springer, Clumber Spaniel, English Setter, Blenheim Spaniel, Border Collie, Garbage Truck, Moving Van, Fire Engine, Ambulance, School Bus, French Horn, Bassoon, Trombone, Oboe, Saxophone, Church, Monastery, Bell Cote, Dome, Library, Cassette Player, Polaroid Camera, Loudspeaker, Typewriter Keyboard, Projector'
        preserved = ' '

    return prompt, preserved

import re
def sanitize_filename(filename):
    # Remove invalid characters and replace spaces with underscores
    # return re.sub(r'[<>:"\\|?*]', '', filename)
    return filename.replace('</w>', '')

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
