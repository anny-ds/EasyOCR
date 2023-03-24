from easydict import EasyDict as edict

config = edict()

config.exp_name = 'Where to store logs and models'
config.train_data = 'path to training dataset'
config.valid_data = 'path to validation dataset'
config.workers = 4          # number of data loading workers
config.manualSeed = 1111     # for random seed setting
config.batch_size = 192     # input batch size
config.num_ite = 300000     # number of iterations to train for
config.valInterval = 2000    # Interval between each validation
config.saved_model = ''         # path to model to continue training
config.FT = True                # whether to do fine-tuning
config.adam = True              # Whether to use adam (default is Adadelta)
config.lr = 1                   # learning rate, default=1.0 for Adadelta
config.beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
config.rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
config.eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
config.grad_clip', type=float, default=5, help='gradient clipping value. default=5')
config.baiduCTC', action='store_true', help='for data_filtering_off mode')

""" Data processing """
config.select_data', type=str, default='MJ-ST',
                    help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
config.batch_ratio', type=str, default='0.5-0.5',
                    help='assign ratio for each selected data in the batch')
config.total_data_usage_ratio', type=str, default='1.0',
                    help='total data usage ratio, this ratio is multiplied to total number of data.')
config.batch_max_length', type=int, default=25, help='maximum-label-length')
config.imgH', type=int, default=32, help='the height of the input image')
config.imgW', type=int, default=100, help='the width of the input image')
config.rgb', action='store_true', help='use rgb input')
config.character', type=str,
                    default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
config.sensitive', action='store_true', help='for sensitive character mode')
config.PAD', action='store_true', help='whether to keep ratio then pad for image resize')
config.data_filtering_off', action='store_true', help='for data_filtering_off mode')
""" Model Architecture """
config.Transformation', type=str, required=True, help='Transformation stage. None|TPS')
config.FeatureExtraction', type=str, required=True,
                    help='FeatureExtraction stage. VGG|RCNN|ResNet')
config.SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
config.Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
config.num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
config.input_channel', type=int, default=1,
                    help='the number of input channel of Feature extractor')
config.output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
config.hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

