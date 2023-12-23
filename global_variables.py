FRACTION_OF_FOOD = 200

# keys of results.json
METHODS = 'methodes'
PERFORMANCES = 'performances'
PERCENTAGE = 'pourcentage'
ACCURACY_TEST = 'accuracy_test'
LOSS_TEST = 'loss_test'
ACCURACY_TRAIN = 'accuracy_train'
LOSS_TRAIN = 'loss_train'
NAME = 'nom'

# names of methods
BASE = 'base'
RANDOM = 'random'
CLUSTERING = 'clustering'
REPRESENTATIVE_SAMPLING = 'representative_sampling'
LEAST_CONFIDENCE = 'least_confidence'
MARGIN = 'margin_of_confidence'
ENTROPY = 'entropy'
MIXED_WITH_LEAST_CONFIDENCE_AND_REPRESENTATIVE_SAMPLING = 'mixed_with_least_confidence_and_representative_sampling'
MIXED_WITH_MARGIN_AND_CLUSTERING = 'mixed_with_margin_and_clustering'

# paths
PRETRAINED_MODEL_PATH = f'models/{BASE}_0.h5'
RESULTS_PATH = 'results/results.json'
FOOD_DATASET_PATH = 'data/ReviewsFood.csv'
CLOTH_DATASET_PATH = 'data/ReviewsCloth.csv'
