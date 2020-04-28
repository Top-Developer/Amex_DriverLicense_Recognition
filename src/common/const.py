# number of model
MODEL_CNT = 5

# define model directory name
MODEL_DIR_PATH = [ 'inference_graph_card_license_28434',  	# model path to detect credit card or driver license
                   'inference_graph_card_info',     		# model path to detect info in credit card and master card
                   'inference_graph_license_info',   		# model path to detect info in driver license
                   'inference_graph_digits_info',			# model path to digits info in credit card and driver license
                   'inference_graph_card_bank_info'			# model path to detect info in credit card and master card
                 ]

# define detection stage index
MODEL_IDX_CARD_LICENSE = 0	    # model index for detection stage of credit card or driver license

# define file name for frozen inference graph
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

# define directory name for label map 
LABEL_MAP_DIR_PATH = [ 'training_card_license',     # labelmap path to detect credit card or driver license
                       'training_card_info',        # labelmap path to detect info in credit card and master card
                       'training_license_info',     # labelmap path to detect info in driver license
                       'training_digits_info',		# labelmap path to detect info in credit card and driver license
                       'training_card_bank_info'	# labelmap path to detect info in credit card and master card
                     ]
                     
# define file name for label map
LABEL_MAP_FILE_NAME = 'labelmap.pbtxt'

# define number of classes
CLASSES_NUM = [ 2,   # number of classes: [credit card, driver license] 
                5,   # number of classes: [card number, card issuing bank, expiration date, cvv code, cardholder name]
                20,  # number of classes: [first name, last name, address, state issuing driver license, issued date,expiration date]
                10,	 # number of classes: [1, 2, ..., 9, 0]
                20,	 # number of classes: [address, expiration date, last name, first name, issued date, state_iowa, state_arizona, ...]
              ]

# define minium score thresh to allow object
MIN_SCORE_THRESH = 0.6   

# define index of model parameters for detection
SESS 			  = 0
DETECTION_BOXES   = 1
DETECTION_SCORES  = 2
DETECTION_CLASSES = 3
CATEGORY_INDEX	  = 4
DETECTED_NUM      = 5
IMG_TENSOR  	  = 6

# define model index matched with class names
MODEL_IDX_FROM_CATEGORY = { 'cvv':3,
						    'card_number':3,
						    'issued_date':3,
						    'expiration_date':3,
						    'issuing_bank':4,
						    'cardholder_name':4
						  }

# define model index
MODEL_IDX_4_DETECT_NUMBER 	 = 3
MODEL_IDX_4_DETECT_BANK_NAME = 4

