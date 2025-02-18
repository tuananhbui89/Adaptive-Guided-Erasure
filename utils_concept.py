import csv 
from imagenet_labels import IMAGENET_1K

class ConceptDict:
    def __init__(self):
        self.all_concepts = {}

    def load_concepts(self, concept_name, csv_file_path):
        
        data = []
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            # next(reader)  # Skip the header
            for row in reader:
                data.append(row[0])

        if concept_name not in self.all_concepts:
            self.all_concepts[concept_name] = []

        self.all_concepts[concept_name].extend(data)

    def load_imagenet_concepts(self):
        # 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias'
        concepts = []
        for index in IMAGENET_1K:
            concept = IMAGENET_1K[index]
            concepts.append(concept.split(',')[0])
        
        self.all_concepts['imagenet'] = concepts


    def get_concepts(self, concept_name):
        if concept_name not in self.all_concepts:
            raise ValueError(f"Concept name '{concept_name}' not found in the dictionary.") 
        return self.all_concepts[concept_name]

    def get_concepts_as_dict(self, concept_name):
        # format: vocab[token] = index
        if concept_name not in self.all_concepts:
            raise ValueError(f"Concept name '{concept_name}' not found in the dictionary.") 
        vocab = {}
        for index, token in enumerate(self.all_concepts[concept_name]):
            vocab[token] = index
        return vocab


    def load_all_concepts(self):
        self.load_concepts('Cassette Player', 'concepts/imagenette_cassette_player.csv')
        self.load_concepts('Chain Saw', 'concepts/imagenette_chain_saw.csv')
        self.load_concepts('Church', 'concepts/imagenette_church.csv')
        self.load_concepts('Gas Pump', 'concepts/imagenette_gas_pump.csv')
        self.load_concepts('Tench', 'concepts/imagenette_tench.csv')
        self.load_concepts('Garbage Truck', 'concepts/imagenette_garbage_truck.csv')
        self.load_concepts('English Springer', 'concepts/imagenette_english_springer.csv')
        self.load_concepts('Golf Ball', 'concepts/imagenette_golf_ball.csv')
        self.load_concepts('parachute', 'concepts/imagenette_parachute.csv')
        self.load_concepts('French Horn', 'concepts/imagenette_french_horn.csv')

        self.load_concepts('imagenette', 'concepts/imagenette_cassette_player.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_chain_saw.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_church.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_gas_pump.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_tench.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_garbage_truck.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_english_springer.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_golf_ball.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_parachute.csv')
        self.load_concepts('imagenette', 'concepts/imagenette_french_horn.csv')

        self.load_concepts('nudity', 'concepts/nudity_human_body.csv')
        self.load_concepts('nudity', 'concepts/nudity_naked_person.csv')
        self.load_concepts('nudity', 'concepts/nudity_nude_person.csv')

        self.load_concepts('human_body', 'concepts/nudity_human_body.csv')

        self.load_concepts('artistic', 'concepts/artistic_concepts.csv')
        self.load_concepts('artistic', 'concepts/artistic_painting.csv')

        self.load_imagenet_concepts()
