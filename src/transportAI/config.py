# =============================================================================
# 1) SETUP
# =============================================================================

#=============================================================================
# 1A) MODULES
#==============================================================================
import os
import numpy as np
import time
import copy


class Config:
    def __init__(self, network_key: str):
        
        self._network_key = network_key

        self._experiment_options = {}
        self._experiment_results = {}

        self.experiment_running = {} # Dictionary of booleans per network
        
        self._sim_options = {}
        self._estimation_options = {}
        self._estimation_results = {}
        self.extra_options = {}
        self._gis_options = {}
        self._gis_results = {}
        self._paths = {}
        self._plots_options = {}
        self._theta_true = {}


        self._log_file = {}

        self.set_config(current_network=network_key)
    
    @property
    def network_key(self):
        return self._network_key

    @network_key.setter
    def network_key(self, value):
        self._network_key = value
        
    @property
    def sim_options(self):
        return self._sim_options

    @sim_options.setter
    def sim_options(self, value):
        self._sim_options = value
        
    @property
    def tntp_networks(self):
        return self._tntp_networks

    @tntp_networks.setter
    def tntp_networks(self, value):
        self._tntp_networks = value
    
    @property
    def experiment_options(self):
        return self._experiment_options

    @experiment_options.setter
    def experiment_options(self, value):
        self._experiment_options = value
        
    @property
    def experiment_results(self):
        return self._experiment_results

    @experiment_results.setter
    def experiment_results(self, value):
        self._experiment_results = value


    @property
    def estimation_options(self):
        return self._estimation_options

    @estimation_options.setter
    def estimation_options(self, value):
        self._estimation_options = value
    
    @property
    def estimation_results(self):
        return self._estimation_results

    @estimation_results.setter
    def estimation_results(self, value):
        self._estimation_results = value
        
    @property
    def log_file(self):
        return self._log_file

    @log_file.setter
    def log_file(self, value):
        self._log_file = value
        
    @property
    def gis_options(self):
        return self._gis_options

    @gis_options.setter
    def gis_options(self, value):
        self._gis_options = value
        
    @property
    def gis_results(self):
        return self._gis_results

    @gis_results.setter
    def gis_results(self, value):
        self._gis_results = value
        
    @property
    def paths(self):
        return self._paths

    @paths.setter
    def paths(self, value):
        self._paths = value

    @property
    def plots_options(self):
        return self._plots_options

    @plots_options.setter
    def plots_options(self, value):
        self._plots_options = value
        
    @property
    def theta_true(self):
        return self._theta_true

    @theta_true.setter
    def theta_true(self, value):
        self._theta_true = value

    def set_log_file(self, networkname):

        current_datetime = time.strftime("%y%m%d %H%M%S")
        self.log_file['dt'] = current_datetime.split(' ')[0]
        self.log_file['ts'] = current_datetime.split(' ')[1]

        subfolder_name = self.log_file['dt'] + '_' + self.log_file['ts']

        self.log_file['folderpath'] = self.paths['output_folder'] + 'log/estimations/' + networkname + '/' + subfolder_name

        # Create a subfolder based on starting date and time of the simulation
        os.mkdir(self.log_file['folderpath'])

    def set_experiments_log_files(self, networkname):

        if networkname not in self.experiment_running.keys():

            current_datetime = time.strftime("%y%m%d %H%M%S")

            self.log_file['dt'] = current_datetime.split(' ')[0]
            self.log_file['ts'] = current_datetime.split(' ')[1]

            subfolder_name = self.log_file['dt'] + '_' + self.log_file['ts']

            self.log_file['folderpath'] = self.paths['output_folder'] + 'log/experiments/' + networkname + '/' + subfolder_name

            # Create a subfolder based on starting date and time of the simulation
            os.mkdir(self.log_file['folderpath'])

            self.experiment_replicate = 1

            self.experiment_running[networkname] = True

        else:

            subsubfolder_name = self.experiment_replicate

            self.log_file['subfolderpath'] = self.log_file['folderpath'] + '/' + str(subsubfolder_name)

            # Create a subfolder based on starting date and time of the simulation
            os.mkdir(self.log_file['subfolderpath'])

            self.experiment_replicate += 1



    def set_custom_networks_matrices(self):

        A, Q = {}, {}

        A['N1'] = np.array([[0, 3], [0, 0]])
        Q['N1'] = np.array([[0, 100], [0, 0]])

        A['N2'] = np.array([[0, 3, 0], [0, 0, 2], [0, 0, 0]])
        Q['N2'] = np.array([[0, 0, 100], [0, 0, 200], [0, 0, 0]])
        # Q['N2'] = Q['N2']/10

        A['N3'] = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0]])
        Q['N3'] = np.array([[0, 0, 0, 100], [0, 0, 0, 200], [0, 0, 0, 300], [0, 0, 0, 0]])
        # Q['N3'] = Q['N3']


        A['N4'] = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]])
        Q['N4'] = np.array([[0, 0, 100, 200], [0, 0, 300, 400], [0, 0, 0, 500], [0, 0, 0, 0]])
        # Q['N4'] = Q['N4']


        # Yang network with no error in OD matrix

        # links_tuples = (1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 8), (5, 9), (6, 9), (7, 8), (
        # 8, 9)
        #
        A['Yang'] = np.array([
            [0, 1, 0, 1, 1, 0, 0, 0,0], [0, 0, 1, 0, 1, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0, 0]
            , [0, 0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]
            , [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        # - True demand martrix
        demand_dict = {(1,6): 120, (1,8): 150, (1,9): 100, (2,6): 130, (2,8): 200, (2,9): 90, (4,6): 80, (4,8): 180, (4,9): 110}

        Q['Yang'] = np.zeros([9,9])

        for od, demand in demand_dict.items():
            Q['Yang'][(od[0]-1,od[1]-1 )] = demand

        # Yang network with error in OD matrix
        A['Yang2'] = copy.deepcopy(A['Yang'])
        Q['Yang2'] = np.zeros([9, 9])

        # - Reference demand matrix (distorted)
        demand_dict = {(1, 6): 100, (1, 8): 130, (1, 9): 120, (2, 6): 120, (2, 8): 170, (2, 9): 140, (4, 6): 110,
                       (4, 8): 170, (4, 9): 105}

        for od, demand in demand_dict.items():
            Q['Yang2'][(od[0]-1,od[1]-1 )] = demand


        # Lo and Chan (2003)

        links_tuples = [(1, 2), (1, 4), (2, 1), (2, 3), (2, 5), (3, 2), (3, 6), (4, 1), (4, 5), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5)]

        A['LoChan'] = np.zeros([6, 6])

        for link_tuple in links_tuples:
            A['LoChan'][(link_tuple[0]-1, link_tuple[1]-1)] = 1

        # - True demand matrix
        demand_dict = {(1,3): 500, (1,4): 250, (1,6): 250, (3,1): 500, (3,4): 250, (3,6): 250, (4,1): 250, (4,3): 500, (4,6): 250, (6,1): 500, (6,3): 250, (6,4): 250 }

        Q['LoChan'] = np.zeros([6, 6])

        for od, demand in demand_dict.items():
            Q['LoChan'][(od[0]-1,od[1]-1 )] = demand



        # Wang et al. (2016)

        links_tuples = [(1, 2), (1, 4), (2, 1), (2, 3), (3, 2), (3, 4), (4, 1), (4, 3)]

        A['Wang'] = np.zeros([4, 4])

        for link_tuple in links_tuples:
            A['Wang'][(link_tuple[0]-1, link_tuple[1]-1)] = 1

        # - True demand matrix
        demand_dict = {(1, 2): 3067, (1, 3): 2489, (1, 4): 4814, (2, 1): 2389, (2, 3): 3774, (2, 4): 1946, (3, 1): 3477, (3, 2): 5772, (3, 4): 4604, (4, 1): 4497, (4, 2): 2773, (4, 3): 4284}

        # TODO: The demand level was reduced for a factor of 10, otherwise, the link capacity would need to be increased by the same amount.

        Q['Wang'] = np.zeros([4, 4])
        for od, demand in demand_dict.items():
            Q['Wang'][(od[0] - 1, od[1] - 1)] = demand/10

        # Store incident matrices
        self.sim_options['custom_networks']['A'] = A
        self.sim_options['custom_networks']['Q'] = Q

    def set_config(self, current_network):

        #=============================================================================
        # 1A) NETWORK SELECTION
        #=============================================================================

        self.sim_options = {}
        # sim_options['current_network'] = ''

        self.sim_options['current_network'] = current_network

        #sim_options['current_network'] = 'Fresno'
        # sim_options['current_network']= 'Sacramento'
        # sim_options['current_network']= 'Colombus'

        # sim_options['current_network'] = 'Barcelona'
        #
        # sim_options['current_network'] = 'SiouxFalls'
        # sim_options['current_network'] = 'Eastern-Massachusetts'
        # sim_options['current_network'] = 'Berlin-Mitte-Center'
        # sim_options['current_network'] = 'Barcelona'


        # Custom networks
        self.custom_networks = ['N1','N2','N3','N4', 'Yang', 'Yang2', 'LoChan', 'Wang']
        self.random_networks = ['N5', 'N6', 'N7', 'N8']


        # List of TNTP networks

        # These networks all work and have less than 1000 links (< 5 minutes)
        self.subfolders_tntp_networks_0 = ['Braess-Example', 'SiouxFalls', 'Eastern-Massachusetts'
            , 'Berlin-Friedrichshain', 'Berlin-Mitte-Center', 'Berlin-Tiergarten', 'Berlin-Prenzlauerberg-Center']

        # Medium size networks that work (< 4000 links) (most take about 15-20 minutes) (record: 3264 links with Terrassa asymmetric)
        self.subfolders_tntp_networks_1 = ['Barcelona', 'Chicago-Sketch', 'Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center',
                                      'Terrassa-Asymmetric', 'Winnipeg', 'Winnipeg-Asymmetric']



        self.tntp_networks = self.subfolders_tntp_networks_0 + self.subfolders_tntp_networks_1

        # Barcelona requires a lot of time (90 minutes) in the logit learning part because there are many od pairs and thus, many paths.

        # # Medium large size networks (5000-10000 links) that have to be tested (they take about 30  minutes)
        # subfolders_tntp_networks = ['Hessen-Asymmetric']

        # Next challenge (or just wait)
        # subfolders_tntp_networks = ['Austin'] # Austin 20000 links, Philadelphia: 40000 links

        # # Too large networks (> 20000 links)
        # subfolders_tntp_networks = ['Berlin-Center'
        #     , 'Birmingham-England', 'chicago-regional', 'Philadelphia','Sydney']
        #
        # # Error in reading data
        # subfolders_tntp_networks = ['Anaheim', 'GoldCoast', 'SymmetricaTestCase]

        # SymmetricaTestCase: Problem because CSV files
        # Anaheim: columns in data indexes
        # Goldcoast: filename

        # Real world networks
        self.real_world_networks = ['Fresno', 'Colombus', 'Sacramento']


        # Working networks
        self.working_networks = self.custom_networks + self.random_networks + self.tntp_networks + self.real_world_networks + ['default']

        assert self.sim_options['current_network'] in self.working_networks, ' unexisting network'

        #=============================================================================
        # 1B) GLOBAL PARAMETERS
        #=============================================================================


        # Network factory
        self.sim_options['n_custom_networks'] = 4 #Do not modify
        self.sim_options['n_random_networks'] = 4
        self.sim_options['q_sparsity'] = 0.99
        self.sim_options['scaling_Q'] = False
        self.sim_options['q_range']= (20, 100) # Range of values on a single cell of the OD matrix of the random networks
        self.sim_options['remove_zeros_Q'] = True #True M and D are smaller which speed up significantly the code

        #range of nodes to create random networks
        self.sim_options['nodes_range'] = 6 + np.arange(1, self.sim_options['n_random_networks'] + 3)

        #IO options
        self.sim_options['reading'] = {'paths': False, 'links': False
            , 'C': False, 'sparse_C': False, 'M': False, 'sparse_M': False, 'D': False, 'sparse_D': False, 'Q': False, 'sparse_Q': False}
        self.sim_options['writing'] = {'paths': False, 'links': False
            , 'C': False, 'sparse_C': False, 'M': False, 'sparse_M': False, 'D': False, 'sparse_D': False, 'Q': False, 'sparse_Q': False}

        # Generation

        # Generation of bpr functions and Z is made at random but using classes of predefined parameters values
        # Generation of Q is made at random but cells are filled with values chosen at random within the q_range
        self.sim_options['generation'] = {'paths': False, 'Q': False, 'bpr': False, 'Z': False, 'C': False, 'M': False, 'D': False, 'links': False, 'fixed_effects': False}

        #Paths

        # Maximum numebr of paths per od for initial paths
        self.sim_options['n_paths'] = 3

        # Maximum number of links for path generation used for simulation and fitting (use more for Fresno or no paths will be foundfor some odswh)
        self.sim_options['cutoff_paths'] = 50

        #If no path is found with the previous cutoff, the path generation method relax the constraint increasing the cutoff by the factor below
        self.sim_options['cutoff_paths_increase_factor'] = 10

        # In every attempt, increase the cutoff by the cutoff factor.
        self.sim_options['max_attempts_path_generation'] = 10

        # Number of paths evaluated within column generation
        self.estimation_options['n_paths_column_generation'] = 0

        # Coverage of OD pairs to sample new paths during column generation
        self.estimation_options['ods_coverage_column_generation'] = 0

        # Number of paths per od to generate synthetic counts
        self.sim_options['n_paths_synthetic_counts'] = 4

        # Value of time
        self.sim_options['vot_hour'] = 10 #USD per hour


        # Attributes included in SUE simulation
        self.estimation_options['k_Z'] = ['s', 'c'] #['wt', 'c', 'length'] # If no constraints are given, then all attribtues are used to compute equilibrium
        self.estimation_options['k_Y'] = ['tt']

        # Preference parameters
        self.theta_true_Y = {}
        self.theta_true_Y['tt'] = -1#-4e-1 #-3e-1

        self.theta_true_Z = {}
        self.theta_true_Z['s'] = -3 #-5e-2
        # self.theta_true_Z['wt'] = 0

        self.theta_true_Z['c'] = self.theta_true_Y['tt'] / (self.sim_options['vot_hour'] / 60) #-1e-2
        # self.theta_true_Z['c'] = 0

        self.theta_true = {'Y': self.theta_true_Y, 'Z': self.theta_true_Z}

        # Initial theta for estimation
        self.theta_0 = dict.fromkeys(self.estimation_options['k_Y']+self.estimation_options['k_Z'],0)

        #Bounds for random search on theta and Q
        self.bounds_theta_0 = {key: (-1, 1) for key, val in self.theta_0.items()}

        self.bounds_q = None # (0, 2)

        # vot = 60 * theta_true_Y['tt'] / theta_true_Z['c']

        # Initial theta search

        self.estimation_options['theta_search'] = None
        self.estimation_options['q_random_search'] = False
        self.estimation_options['n_draws_random_search'] = 0
        self.estimation_options['scaling_Q'] = False


        # Mean count used for benchmark model (if none, it computes the mean of the observed counts)
        self.estimation_results['mean_count_benchmark_model'] = None

        # Additional features


        # self.sim_options['ue_factor'] = 1 # As higher is this factor, all or nothing behavior appears and become hard to identify travellers preferences.

        # Scaling
        # scale_features = {'mean': False, 'std': False}
        # scale_features = {'mean': True, 'std': True}
        self.estimation_options['scale_features'] = {'mean': False, 'std': False}

        # Simulated or real counts (relevant for Fresno case)

        # sim_options['simulated_counts'] = simulated_counts
        # if sim_options['current_network'] == 'Fresno':
        #     sim_options['simulated_counts'] = False

        self.sim_options['simulated_counts'] = False

        # Congested or uncongested mode
        self.sim_options['uncongested_mode'] = False
        # * In uncongested mode only 1 MSA iteration is made. More iterations are useful for the congested case
        # * In uncongested mode, the parameters of the bpr functions are set equal to 0 so the link travel time is just free flow travel time

        #known path set
        self.estimation_options['known_pathset_mode'] = False

        # Out of sample prediction mode
        self.estimation_options['outofsample_prediction_mode'] = False

        # Link coverage
        self.sim_options['max_link_coverage'] = 1

        # Iterations in MSA to generate simulated counts
        self.sim_options['max_sue_iters'] = 100
        # * For the simulated case, 100 has been shown to ensure good convergence for equilibrium, which is key for the bilevel opt as well)

        # - Variance of lognormal distribution or Poisson in noise in OD matrix
        self.sim_options['noise_params']= {'sd_Q': 0.0}
        # sim_options['noise_params'].update({'sd_Q': 500})
        # sim_options['noise_params'].update({'sd_Q': 'Poisson'})

        # Notes: with poisson the noise is litte and there is almost no impact on convergence.
        # No need to set parameters for distribution as they are determined by mean of non-zero entries of OD matrix

        # - Scaling of Q matrix (1 is the default)
        self.sim_options['noise_params'].update({'scale_Q': 1})
        # sim_options['noise_params'].update({'scale_Q': 0.5})

        # Notes: In an uncongested network the scale of Q does not matter

        # Fixed effect by link or OD zone
        self.sim_options['fixed_effects'] = {'Q': False, 'nodes': False, 'coverage': 0}

        # Regularization parameters

        #- Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
        self.sim_options['n_R'] = 1#2 #5 #10 #20 #50

        #Labels of sparse attributes
        self.sim_options['R_labels'] = ['k' + str(i) for i in np.arange(0, self.sim_options['n_R'])]

        #Regularization options
        self.sim_options['prop_validation_sample'] = 0
        self.sim_options['regularization'] = False




        # Experiments
        self.experiment_options['experiment_mode'] = None
        self.experiment_options['monotonicity_experiment'] = False
        self.experiment_options['pseudoconvexity_experiment'] = False
        self.experiment_options['optimization_benchmark_experiment'] = False
        self.experiment_options['convergence_experiment'] = False
        self.experiment_options['consistency_experiment'] = False
        self.experiment_options['irrelevant_attributes_experiment'] = False
        self.experiment_options['noisy_od_counts_experiment'] = False
        self.experiment_options['Yang_biased_reference_od_experiment'] = False
        self.experiment_options['inference_experiment'] = False
        self.experiment_options['inductive_bias_experiment'] = False

        # =============================================================================
        # 4) ESTIMATION OPTIONS
        # =============================================================================

        # self.estimation_options = {}

        # i) For fresno

        if self.sim_options['current_network'] == 'Fresno':
            # Iterations of bilevel problem in refined and no-refined stages
            self.estimation_options['bilevel_iters_regularized'] = 2  # 10
            self.estimation_options['bilevel_iters_norefined'] = 10 #10
            self.estimation_options['bilevel_iters_refined'] = 10 #5
            # * 10 in each works good for simulated networks

            # Iterations of optimization algorithm in outer problem
            self.estimation_options['iters_regularized'] = 1
            self.estimation_options['iters_norefined'] = 1#1 # ngd iterations
            self.estimation_options['iters_refined'] = 1 #1 # gauss newton iterations
            # * The complexity of the ngd iteration is proportional to the number of attribtues since a loop is done over attribute types
            # * ngd iterations are very expensive to compute for large scale network (10 of overhead plus 10-15 secs extra for each attribute in Fresno) so it is done only once with multiple attrs


            # Equilibrium iteration in refined and no refined stages
            self.estimation_options['max_sue_iters_regularized'] = 10  # 20
            self.estimation_options['max_sue_iters_norefined'] = 20#20
            self.estimation_options['max_sue_iters_refined'] = 50 #100 #
            # * Each msa iteration takes about 10 seconds for Fresn

            # Learning rate for first order optimization methods
            self.estimation_options['eta_regularized'] = 1e-2
            self.estimation_options['eta_norefined'] = 1e-2
            self.estimation_options['eta_refined'] = 1e-3
            #* If a larger learning rate is used ( > 1e-3), the optimization may start to bump in.
            # Attributes should be rescaled to have a more robust learning rate.


        # ii) For simulated networks
        else:

            self.estimation_options['bilevel_iters_regularized'] = 5  # 10
            self.estimation_options['bilevel_iters_norefined'] = 10
            self.estimation_options['bilevel_iters_refined'] = 10# 10
            # * 10 in each works good for simulated networks

            # Iterations of optimization algorithm in outer problem
            self.estimation_options['iters_regularized'] = 1
            self.estimation_options['iters_norefined'] = 1 #5
            self.estimation_options['iters_refined'] = 1 # gauss newton
            # * 5 ngd iterations works well for simulated network and under congested mode. For consistency with real network, only 1 is used

            # Equilibrium iteration in refined and no refined stages
            self.estimation_options['max_sue_iters_regularized'] = 20  # 20
            self.estimation_options['max_sue_iters_norefined'] = 20
            self.estimation_options['max_sue_iters_refined'] = 20 #if

            # * In the refined stage and with simulated data, the msa iterations are set equal to the iterations used to produce the counts (100)
            # * In the non-refined stage, it works well to use a 5% of the iteration in refined stage, np.ceil(estimation_options['max_sue_iters_refined']/20)
            #
            # Learning rate for ngd
            self.estimation_options['eta_regularized'] = 5e-2
            self.estimation_options['eta_norefined'] =5e-1
            self.estimation_options['eta_refined'] = 1e-3

            #1e-2 works well for simulated networks without noise.
            # 5e-3 works well for simulated networks with noise.
            # As higher the noise, a lower learning rate works better

        # Initial value set for logit parameters

        # =============================================================================
        # 4) LINKS ATTRIBUTES
        # =============================================================================

        # Define multiple BPR instances/classes that will affect the travel time solution at equilibrium
        self.sim_options['bpr_classes'] = {}
        self.sim_options['bpr_classes']['1'] = {'alpha':0.15, 'beta':4, 'tf':1e-1, 'k':1800}
        self.sim_options['bpr_classes']['2'] = {'alpha':0.15, 'beta':4, 'tf':2e-1, 'k':1800}
        self.sim_options['bpr_classes']['3'] = {'alpha':0.15, 'beta':4, 'tf':3e-1, 'k':1800}
        self.sim_options['bpr_classes']['4'] = {'alpha':0.15, 'beta':4, 'tf':4e-1, 'k':1800}
        self.sim_options['bpr_classes']['5'] = {'alpha':0.15, 'beta':4, 'tf':5e-1, 'k':1800}
        # self.sim_options['bpr_classes']['1'] = {'alpha': 0.15, 'beta': 4, 'tf': 1e0, 'k': 1800}
        # self.sim_options['bpr_classes']['2'] = {'alpha': 0.15, 'beta': 4, 'tf': 2e0, 'k': 1800}
        # self.sim_options['bpr_classes']['3'] = {'alpha': 0.15, 'beta': 4, 'tf': 3e0, 'k': 1800}
        # self.sim_options['bpr_classes']['4'] = {'alpha': 0.15, 'beta': 4, 'tf': 4e0, 'k': 1800}
        # self.sim_options['bpr_classes']['5'] = {'alpha': 0.15, 'beta': 4, 'tf': 5e0, 'k': 1800}
        # sim_options['bpr_classes']['1'] = BPR(alpha=3E-0, beta=1, tf=10E-2, k=1E-0)
        # sim_options['bpr_classes']['2'] = BPR(alpha=2E-0, beta=2, tf=40E-2, k=1E-0)
        # sim_options['bpr_classes']['3'] = BPR(alpha=5E-0, beta=1, tf=33E-2, k=1E-0)
        # sim_options['bpr_classes']['4'] = BPR(alpha=5E-0, beta=2, tf=6E-2, k=1E-0)
        # sim_options['bpr_classes']['5'] = BPR(alpha=2E-0, beta=1, tf=10E-2, k=1E-0)


        # Attribute levels in matrix Z which are non-zero
        self.sim_options['Z_attrs_classes'] = {}
        #  - Waiting time (minutes)
        self.sim_options['Z_attrs_classes']['s'] = dict({'1': 3, '2': 2, '3': 1, '4': 1, '5': 5, '6': 3})
        #  - Cost in USD (e.g. toll)
        self.sim_options['Z_attrs_classes']['c'] = dict({'1': 1, '2': 2, '3': 1.5, '4': 1, '5': 3})



        # =============================================================================
        # 5) CUSTOM NETWORKS ATTRIBUTES
        # =============================================================================

        self.sim_options['custom_networks'] = {}
        self.sim_options['custom_networks']['A'] = {}
        self.sim_options['custom_networks']['Q'] = {}

        self.set_custom_networks_matrices()


        # =============================================================================
        # 4) GIS
        # =============================================================================

        self.gis_options['crs_ca'] = 2228
        self.gis_options['crs_mercator'] = 4326

        self.gis_options['inrix_matching'] = {}
        self.gis_options['data_processing'] = {}
        self.gis_options['buffer_size'] = {}

        self.gis_results['matching_stats'] = {}


        #============================================================================
        # 3) PATHS AND PLOTS
        #=============================================================================

        # Set working directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # Paths to folders or files
        self.paths = {}

        # Github networks path folders
        self.paths['folder_tntp_networks'] = os.getcwd() + "/input/public/networks/github/"
        self.paths['folder_pablo_networks'] = os.getcwd() + "/input/public/networks/pablo/"

        # Colombus Ohio network
        self.paths['Colombus_network'] = "/Users/pablo/google-drive/university/cmu/2-research/datasets/private/bin-networks/Columbus"

        # Fresno and Sac
        self.paths['Fresno_network'] = '/Users/pablo/google-drive/university/cmu/2-research/datasets/private/od-fresno-sac/SR41'
        self.paths['Sacramento_network'] = '/Users/pablo/google-drive/university/cmu/2-research/datasets/private/od-fresno-sac/sac'
        self.paths['Fresno_Sac_networks'] = '/Users/pablo/google-drive/university/cmu/2-research/datasets/private/od-fresno-sac'

        # Folder to write network data
        self.paths['network_data'] = os.getcwd() + "/output/network-data/"

        # Folder to write data
        self.paths['output_folder'] = os.getcwd() + "/output/"

        # Folder to read data
        self.paths['input_folder'] = os.getcwd() + "/input/"

        # Visualization
        self.plots_options = {}
        self.plots_options['dim_subplots'] = int(np.ceil(np.sqrt(self.sim_options['n_random_networks'] + self.sim_options['n_custom_networks']))), int(np.ceil(np.sqrt(self.sim_options['n_random_networks'] + self.sim_options['n_custom_networks'])))
        # plots_options['folder_plots'] = os.getcwd() + '/examples' + '/plots'
        self.plots_options['folder_plots'] = os.getcwd() + '/output' + '/plots'


        #Store dictionaries in self.config object

        self._sim_options = self.sim_options.copy()
        self._estimation_options = self.estimation_options.copy()
        self._gis_options = self.gis_options.copy()
        self._paths = self.paths.copy()
        self._plots_options = self.plots_options.copy()
        self._theta_true = self.theta_true.copy()

    def set_consistency_experiment(self, replicates, theta_0_range, uncongested_mode, consistency_experiment = True, iters_factor = 1, sd_x = 0.01):

        if consistency_experiment:

            print('\nEnabling consistency experiment')

            self.experiment_options['experiment_mode'] = 'consistency experiment'

            # -ii) Noise in link and OD matrix
            self.experiment_options['consistency_experiment'] = True

            # Note: the number represent the number of repllicates for the noise experiments. If 0, no experiment is performed
            self.experiment_options['replicates'] = replicates

            # Range of values used for the random initialization of the parameters
            self.experiment_options['theta_0_range'] = theta_0_range

            self.experiment_options['current_network'] = self.sim_options['current_network']

            # Optimization methods used in no refined and refined stages
            self.experiment_options['outeropt_method_norefined'] = 'ngd'  # 'adam'

            # self.experiment_options['outeropt_method_refined'] = 'gauss-newton'
            self.experiment_options['outeropt_method_refined'] = 'lm'

            # self.experiment_options['outeropt_method_combined'] = 'gauss-newton'
            self.experiment_options['outeropt_method_combined'] = 'lm'


            # defaults

            self.experiment_options['n_paths_synthetic_counts'] = self.sim_options['n_paths_synthetic_counts']

            self.experiment_options['noise_params'] = {}
            self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': sd_x})
            # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
            self.experiment_options['noise_params'].update({'sd_Q': 0})
            self.experiment_options['noise_params'].update({'scale_Q': 1})

            self.experiment_options['accuracy_eq'] = self.estimation_options['accuracy_eq']
            self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']

            self.experiment_options['max_sue_iters'] =  self.sim_options['max_sue_iters']
            self.experiment_options['max_sue_iters_norefined'] = 50  # 1
            self.experiment_options['max_sue_iters_refined'] = 50  # 5 #

            self.experiment_options['iters_norefined'] = self.estimation_options['iters_norefined']
            self.experiment_options['iters_refined'] = self.estimation_options['iters_refined']

            self.experiment_options['eta_norefined'] = 1e-1 #1e-0
            self.experiment_options['eta_refined'] = 1e-1
            # Note: increasing the number of gradient updates in ngd is the best way to go. As more noise, more gradient updates should be done

            # The iterations of NGD distributed between bilevel iters and gradient update must be at least 10
            # self.experiment_options['bilevel_iters_norefined'] = 10
            self.experiment_options['bilevel_iters_norefined'] = 10*iters_factor  # 5
            self.experiment_options['bilevel_iters_refined'] = 20*iters_factor  # 10
            # Additional iterations with gn when using combined algorithm
            self.experiment_options['bilevel_iters_combined'] = self.experiment_options['bilevel_iters_refined'] - self.experiment_options['bilevel_iters_norefined']

            # # Do 40 % less so the copmutatino time with refined iterations only gets equated
            # self.experiment_options['bilevel_iters_combined'] = int(
            #     0.6 * self.experiment_options['bilevel_iters_combined'])


            if uncongested_mode:

                self.set_uncongested_mode(True)
                self.experiment_options['uncongested_mode'] = True
                self.experiment_options['max_sue_iters'] = 0
                self.experiment_options['max_sue_iters_norefined'] = 0
                self.experiment_options['max_sue_iters_refined'] = 0

            else:
                self.set_uncongested_mode(False)
                self.experiment_options['uncongested_mode'] = False

    def set_irrelevant_attributes_experiment(self, replicates, theta_0_range, uncongested_mode, consistency_experiment = True, iters_factor = 1,sd_x = 0.01):

        if consistency_experiment:

            print('\nEnabling irrelevant attributes experiment')

            self.experiment_options['experiment_mode'] = 'irrelevant_attributes_experiment'

            # -ii) Noise in link and OD matrix
            self.experiment_options['irrelevant_attributes_experiment'] = True

            # Note: the number represent the number of repllicates for the noise experiments. If 0, no experiment is performed
            self.experiment_options['replicates'] = replicates

            # Range of values used for the random initialization of the parameters
            self.experiment_options['theta_0_range'] = theta_0_range

            self.experiment_options['current_network'] = self.sim_options['current_network']

            # Optimization methods used in no refined and refined stages
            self.experiment_options['outeropt_method_norefined'] = 'ngd'  # 'adam'
            # self.experiment_options['outeropt_method_refined'] = 'gauss-newton'  # 'lm' #gauss-newton

            # self.experiment_options['outeropt_method_refined'] = 'gauss-newton'
            self.experiment_options['outeropt_method_refined'] = 'lm'

            self.experiment_options['outeropt_method_combined'] = 'ngd'
            # self.experiment_options['outeropt_method_combined'] = 'gauss-newton'

            # defaults

            self.experiment_options['n_paths_synthetic_counts'] = self.sim_options['n_paths_synthetic_counts']

            self.experiment_options['noise_params'] = {}
            self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': sd_x})
            # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
            self.experiment_options['noise_params'].update({'sd_Q': 0})
            self.experiment_options['noise_params'].update({'scale_Q': 1})

            self.experiment_options['accuracy_eq'] = self.estimation_options['accuracy_eq']
            self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']

            self.experiment_options['max_sue_iters'] =  self.sim_options['max_sue_iters']
            self.experiment_options['max_sue_iters_norefined'] = 50  # 1
            self.experiment_options['max_sue_iters_refined'] = 50  # 5 #

            self.experiment_options['iters_norefined'] = self.estimation_options['iters_norefined']
            self.experiment_options['iters_refined'] = self.estimation_options['iters_refined']

            self.experiment_options['eta_norefined'] = 1e-1 #1e-0
            self.experiment_options['eta_refined'] = 1e-2
            self.experiment_options['eta_combined'] = 1e-2
            # Note: increasing the number of gradient updates in ngd is the best way to go. As more noise, more gradient updates should be done

            # The iterations of NGD distributed between bilevel iters and gradient update must be at least 10
            # self.experiment_options['bilevel_iters_norefined'] = 10
            self.experiment_options['bilevel_iters_norefined'] = 10*iters_factor  # 5
            self.experiment_options['bilevel_iters_refined'] = 10*iters_factor  # 10
            # Additional iterations with gn when using combined algorithm
            self.experiment_options['bilevel_iters_combined'] = 10*iters_factor

            # # Do 40 % less so the copmutatino time with refined iterations only gets equated
            # self.experiment_options['bilevel_iters_combined'] = int(
            #     0.6 * self.experiment_options['bilevel_iters_combined'])


            if uncongested_mode:

                self.set_uncongested_mode(True)
                self.experiment_options['uncongested_mode'] = True
                self.experiment_options['max_sue_iters'] = 0
                self.experiment_options['max_sue_iters_norefined'] = 0
                self.experiment_options['max_sue_iters_refined'] = 0

            else:
                self.set_uncongested_mode(False)
                self.experiment_options['uncongested_mode'] = False
                self.experiment_options['max_sue_iters'] = 50
                self.experiment_options['max_sue_iters_norefined'] = 50
                self.experiment_options['max_sue_iters_refined'] = 50

    def set_noisy_od_counts_experiment(self, replicates, theta_0_range, uncongested_mode, sds_x: list = [], sds_Q: list = [], scales_Q: list = [], coverages: list = [], sd_x = 0.01, sd_Q = 0, scale_Q = 1, run_experiment = True, iters_factor = 1):

        if run_experiment:

            print('\nEnabling noisy od/counts experiment')

            self.experiment_options['experiment_mode'] = 'noisy od/counts experiment'

            # -ii) Noise in link and OD matrix
            self.experiment_options['noisy_od_counts_experiment'] = True

            # Note: the number represent the number of repllicates for the noise experiments. If 0, no experiment is performed
            self.experiment_options['replicates'] = replicates

            # Range of values used for the random initialization of the parameters
            self.experiment_options['theta_0_range'] = theta_0_range

            self.experiment_options['current_network'] = self.sim_options['current_network']

            self.experiment_options['n_paths_synthetic_counts'] = self.sim_options['n_paths_synthetic_counts']

            self.experiment_options['noise_params'] = {}
            self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': sd_x})
            # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
            self.experiment_options['noise_params'].update({'sd_Q': sd_Q})
            self.experiment_options['noise_params'].update({'scale_Q': scale_Q})

            self.experiment_options['sds_x'] = sds_x
            self.experiment_options['sds_Q'] = sds_Q
            self.experiment_options['scales_Q'] = scales_Q
            self.experiment_options['coverages'] = coverages

            self.experiment_options['accuracy_eq'] = self.estimation_options['accuracy_eq']
            self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']

            # Optimization methods used in no refined and refined stages
            self.experiment_options['outeropt_method_norefined'] = 'ngd'  # 'adam'
            # self.experiment_options['outeropt_method_refined'] = 'gauss-newton'
            self.experiment_options['outeropt_method_refined'] = 'ngd'

            self.experiment_options['eta_norefined'] = 1e-1  # 1e-0
            self.experiment_options['eta_refined'] = 1e-2


            self.experiment_options['max_sue_iters'] =  self.sim_options['max_sue_iters']
            self.experiment_options['max_sue_iters_norefined'] = 50  # 1
            self.experiment_options['max_sue_iters_refined'] = 50  # 5 #

            self.experiment_options['iters_norefined'] = self.estimation_options['iters_norefined']
            self.experiment_options['iters_refined'] = self.estimation_options['iters_refined']

            # Note: increasing the number of gradient updates in ngd is the best way to go. As more noise, more gradient updates should be done

            # The iterations of NGD distributed between bilevel iters and gradient update must be at least 10
            # self.experiment_options['bilevel_iters_norefined'] = 10
            self.experiment_options['bilevel_iters_norefined'] = 10*iters_factor  # 5
            self.experiment_options['bilevel_iters_refined'] = 0*10*iters_factor  # 10
            # Additional iterations with gn when using combined algorithm
            self.experiment_options['bilevel_iters_combined'] = self.experiment_options['bilevel_iters_refined'] - self.experiment_options['bilevel_iters_norefined']

            # # Do 40 % less so the copmutatino time with refined iterations only gets equated
            # self.experiment_options['bilevel_iters_combined'] = int(
            #     0.6 * self.experiment_options['bilevel_iters_combined'])

            if uncongested_mode:

                self.set_uncongested_mode(True)
                self.experiment_options['uncongested_mode'] = True
                self.experiment_options['max_sue_iters'] = 0
                self.experiment_options['max_sue_iters_norefined'] = 0
                self.experiment_options['max_sue_iters_refined'] = 0

            else:
                self.set_uncongested_mode(False)
                self.experiment_options['uncongested_mode'] = False
                self.experiment_options['max_sue_iters'] = 50
                self.experiment_options['max_sue_iters_norefined'] = 50
                self.experiment_options['max_sue_iters_refined'] = 53

    # def set_noise_experiment(self, replicates, theta_0_range, sd_x = 0, sd_Q = 0, scale_Q = 1):
    #     """ Wrapper function to perform noise experiment but without noise """
    #
    #     self.experiment_options['consistency_experiment'] = True
    #
    #     self.set_consistency_experiment(noise_experiment = True, replicates = replicates
    #                                     , theta_0_range = theta_0_range, sd_x=0, sd_Q=0, scale_Q=1)

    def set_optimization_benchmark_experiment(self, optimization_benchmark, replicates, theta_0_range, uncongested_mode):

        if optimization_benchmark:

            print('\nEnabling optimization benchmark experiment')

            self.experiment_options['experiment_mode'] = 'optimization benchmark'
            self.experiment_options['optimization_benchmark_experiment'] = True

            # Note: the number represent the number of repllicates for the noise experiments. If 0, no experiment is performed
            self.experiment_options['replicates'] = replicates

            # Range of values used for to generate the grid of values used for the initialization of the parameters
            self.experiment_options['theta_0_range'] = theta_0_range

            self.experiment_options['current_network'] = self.sim_options['current_network']

            # defaults

            self.experiment_options['n_paths_synthetic_counts'] = self.sim_options['n_paths_synthetic_counts']

            self.experiment_options['noise_params'] = {}
            self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': 0})
            # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
            self.experiment_options['noise_params'].update({'sd_Q': 0})
            self.experiment_options['noise_params'].update({'scale_Q': 1})

            self.experiment_options['accuracy_eq'] = self.estimation_options['accuracy_eq']
            self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']

            self.experiment_options['max_sue_iters'] = self.sim_options['max_sue_iters']

            self.experiment_options['iters'] = self.estimation_options['iters_norefined']

            # self.experiment_options['eta'] = 3e-1
            self.experiment_options['eta'] = 1e-0
            # Note: increasing the number of gradient updates in ngd is the best way to go. As more noise, more gradient updates should be done

            # The iterations of NGD distributed between bilevel iters and gradient update must be at least 10
            self.experiment_options['bilevel_iters'] = 2  # 5

            if uncongested_mode:

                self.set_uncongested_mode(True)
                self.experiment_options['uncongested_mode'] = True
                self.experiment_options['max_sue_iters'] = 0
                self.experiment_options['max_sue_iters_norefined'] = 0
                self.experiment_options['max_sue_iters_refined'] = 0

            else:
                self.set_uncongested_mode(False)
                self.experiment_options['uncongested_mode'] = False
                self.experiment_options['max_sue_iters'] = 50
                self.experiment_options['max_sue_iters_norefined'] = 50
                self.experiment_options['max_sue_iters_refined'] = 50


    def set_monotonicity_experiment(self, theta_grid, uncongested_mode, run_experiment = True):

        if not run_experiment:
            return None

        print('\nEnabling monotonocity experiment')

        self.experiment_options['current_network'] = self.sim_options['current_network']

        self.experiment_options['monotonicity_experiment'] = True

        self.experiment_options['experiment_mode'] = 'monotonocity'


        self.experiment_options['noise_params'] = {}
        self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': 0})
        # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
        self.experiment_options['noise_params'].update({'sd_Q': 0})
        self.experiment_options['noise_params'].update({'scale_Q': 1})

        # self.experiment_options['grid_search'] = {}
        self.experiment_options['theta_grid'] = theta_grid

        #defaults
        self.experiment_options['n_paths_synthetic_counts'] = self.sim_options['n_paths_synthetic_counts']
        self.experiment_options['max_sue_iters'] = self.sim_options['max_sue_iters']
        self.experiment_options['accuracy_eq'] = self.estimation_options['accuracy_eq']
        self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']
        # self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']

        if uncongested_mode:

            self.set_uncongested_mode(True)
            self.experiment_options['uncongested_mode'] = True
            self.experiment_options['max_sue_iters'] = 0
            self.experiment_options['max_sue_iters_norefined'] = 0
            self.experiment_options['max_sue_iters_refined'] = 0

        else:
            self.set_uncongested_mode(False)
            self.experiment_options['uncongested_mode'] = False
            self.experiment_options['max_sue_iters'] = 50
            self.experiment_options['max_sue_iters_norefined'] = 50
            self.experiment_options['max_sue_iters_refined'] = 50

    def set_inductive_bias_experiment(self, uncongested_mode, run_experiment = True):

        if not run_experiment:
            return None


        self.experiment_options['inductive_bias_experiment'] = True

        self.experiment_options['experiment_mode'] = 'inductive_bias_experiment'

        # Initial parameter values (from Yang are 14, 10, 4.2)
        # config.theta_0 = dict.fromkeys(config.theta_0, -2)
        # config.theta_0 = dict.fromkeys(config.theta_0, -4.2)
        self.experiment_options['theta_0'] = dict.fromkeys({'tt'}, -14)

        if uncongested_mode:

            self.set_uncongested_mode(True)
            self.experiment_options['uncongested_mode'] = True
            self.experiment_options['max_sue_iters'] = 0
            self.experiment_options['max_sue_iters_norefined'] = 0
            self.experiment_options['max_sue_iters_refined'] = 0

        else:
            self.set_uncongested_mode(False)
            self.experiment_options['uncongested_mode'] = False
            self.experiment_options['max_sue_iters'] = 50
            self.experiment_options['max_sue_iters_norefined'] = 50
            self.experiment_options['max_sue_iters_refined'] = 50

    def set_inference_experiment(self, uncongested_mode, sd_x, run_experiment=True):

        if not run_experiment:
            return None

        self.experiment_options['inference_experiment'] = True

        self.experiment_options['experiment_mode'] = 'inference_experiment'

        self.experiment_options['noise_params'] = {}
        self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': sd_x})
        # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
        self.experiment_options['noise_params'].update({'sd_Q': 0})
        self.experiment_options['noise_params'].update({'scale_Q': 1})

        # Initial parameter values (from Yang are 14, 10, 4.2)
        # config.theta_0 = dict.fromkeys(config.theta_0, -2)
        # config.theta_0 = dict.fromkeys(config.theta_0, -4.2)
        self.experiment_options['theta_0'] = dict.fromkeys({'tt'}, -14)

        if uncongested_mode:

            self.set_uncongested_mode(True)
            self.experiment_options['uncongested_mode'] = True
            self.experiment_options['max_sue_iters'] = 0
            self.experiment_options['max_sue_iters_norefined'] = 0
            self.experiment_options['max_sue_iters_refined'] = 0

        else:
            self.set_uncongested_mode(False)
            self.experiment_options['uncongested_mode'] = False
            self.experiment_options['max_sue_iters'] = 50
            self.experiment_options['max_sue_iters_norefined'] = 50
            self.experiment_options['max_sue_iters_refined'] = 50

    def set_od_bias_yang_experiment(self, uncongested_mode, run_experiment = True):

        if not run_experiment:
            return None

        self.experiment_options['Yang_biased_reference_od_experiment'] = True

        # Initial parameter values (from Yang are 14, 10, 4.2)
        # config.theta_0 = dict.fromkeys(config.theta_0, -2)
        # config.theta_0 = dict.fromkeys(config.theta_0, -4.2)
        self.experiment_options['theta_0'] = dict.fromkeys({'tt'}, -14)


        self.experiment_options['experiment_mode'] = 'Yang biased reference od experiment'

        # Learning rate for first order optimization
        self.experiment_options['eta_norefined'] = 1e-0
        # config.experiment_options['eta_norefined'] = 3e-0
        self.experiment_options['eta_refined'] = 1e-3

        if uncongested_mode:

            self.set_uncongested_mode(True)
            self.experiment_options['uncongested_mode'] = True
            self.experiment_options['max_sue_iters'] = 0
            self.experiment_options['max_sue_iters_norefined'] = 0
            self.experiment_options['max_sue_iters_refined'] = 0

        else:
            self.set_uncongested_mode(False)
            self.experiment_options['uncongested_mode'] = False
            self.experiment_options['max_sue_iters'] = 50
            self.experiment_options['max_sue_iters_norefined'] = 50
            self.experiment_options['max_sue_iters_refined'] = 50
            
    def set_convergence_experiment(self, iters_factor = 1, uncongested_mode= True, run_experiment = True):

        if not run_experiment:
            return None

        self.experiment_options['convergence_experiment'] = True
        self.experiment_options['experiment_mode'] = 'Convergence experiment'
        
        #Optimization options
        self.experiment_options['outeropt_method_norefined'] = 'ngd'  # 'adam'
        self.experiment_options['outeropt_method_refined'] = 'gauss-newton'
        # self.experiment_options['outeropt_method_refined'] = 'lm' #'lm' #gauss-newton
        # config.experiment_options['outeropt_method_refined'] = 'ngd' #'adam'

        # Learning rate for first order optimization
        self.experiment_options['eta_norefined'] = 1e-2
        # self.experiment_options['eta_norefined'] = 5e-2
        # config.experiment_options['eta_norefined'] = 3e-0
        self.experiment_options['eta_refined'] = 1e-3

        self.experiment_options['bilevel_iters_norefined'] = iters_factor*10  # 10
        self.experiment_options['bilevel_iters_refined'] = iters_factor*10  # 5

        if uncongested_mode:

            self.set_uncongested_mode(True)
            self.sim_options['uncongested_mode'] = True
            self.experiment_options['uncongested_mode'] = True
            self.experiment_options['max_sue_iters'] = 0
            self.experiment_options['max_sue_iters_norefined'] = 0
            self.experiment_options['max_sue_iters_refined'] = 0

        else:
            self.set_uncongested_mode(False)
            self.sim_options['uncongested_mode'] = False
            self.experiment_options['uncongested_mode'] = False
            self.experiment_options['max_sue_iters'] = 50
            self.experiment_options['max_sue_iters_norefined'] = 50
            self.experiment_options['max_sue_iters_refined'] = 50

    def set_pseudoconvexity_experiment(self, theta_grid, uncongested_mode, grid_search = True):

        if grid_search:
            print('\nEnabling pseudo-convexity experiment')

            self.experiment_options['current_network'] = self.sim_options['current_network']

            self.experiment_options['pseudoconvexity_experiment'] = True

            self.experiment_options['experiment_mode'] = 'grid search'


            self.experiment_options['noise_params'] = {}
            self.experiment_options['noise_params'].update({'mu_x': 0, 'sd_x': 0})
            # self.experiment_options['noise_params'].update({'sd_Q': 'Poisson'})
            self.experiment_options['noise_params'].update({'sd_Q': 0})
            self.experiment_options['noise_params'].update({'scale_Q': 1})

            # self.experiment_options['grid_search'] = {}
            self.experiment_options['theta_grid'] = theta_grid

            #defaults
            self.experiment_options['n_paths_synthetic_counts'] = self.sim_options['n_paths_synthetic_counts']
            self.experiment_options['max_sue_iters'] = self.sim_options['max_sue_iters']
            self.experiment_options['accuracy_eq'] = self.estimation_options['accuracy_eq']
            self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']
            # self.experiment_options['max_link_coverage'] = self.sim_options['max_link_coverage']

            if uncongested_mode:

                self.set_uncongested_mode(True)
                self.experiment_options['uncongested_mode'] = True
                self.experiment_options['max_sue_iters'] = 0
                self.experiment_options['max_sue_iters_norefined'] = 0
                self.experiment_options['max_sue_iters_refined'] = 0

            else:
                self.set_uncongested_mode(False)
                self.experiment_options['uncongested_mode'] = False
                self.experiment_options['max_sue_iters'] = 50
                self.experiment_options['max_sue_iters_norefined'] = 50
                self.experiment_options['max_sue_iters_refined'] = 50


    def set_uncongested_mode(self, uncongested: bool = True):



        self.sim_options['uncongested_mode'] = uncongested

        if self.sim_options['uncongested_mode']:

            print('\nEnabling uncongested mode')

            # In ncongested case it is sufficnet to set the iterations of equilibrium to be equal to 1.
            self.estimation_options['max_sue_iters_regularized'] = 0  # 100
            self.estimation_options['max_sue_iters_norefined'] = 0  # 100

            # Given that it is very expensive to compute path probabitilies and the resuls are good already, it seems fine to perform only one iteration for outer problem
            self.estimation_options['max_sue_iters_refined'] = 0  # 5

            # We also set the iterations to generate traffic counts to 1
            self.sim_options['max_sue_iters'] = 0  # 100


    def set_known_pathset_mode(self, known: bool = False):

        print('\nEnabling known path set mode')

        self.estimation_options['known_pathset_mode'] = known

        if self.estimation_options['known_pathset_mode']:
            self.estimation_options['n_paths_column_generation'] = 0
            self.estimation_options['k_path_set_selection'] = 0
            self.estimation_options['ods_coverage_column_generation'] = 0

            # If none, the initial path set is used to generate counts
            self.sim_options['n_paths_synthetic_counts'] = None


    def set_outofsample_prediction_mode(self, theta: dict, mean_count, outofsample_prediction: bool = False):

        print('\nEnabling out of sample prediction mode')

        self.estimation_options['outofsample_prediction_mode'] = outofsample_prediction

        if self.estimation_options['outofsample_prediction_mode']:

            self.estimation_options['bilevel_iters_norefined'] = 1
            self.estimation_options['bilevel_iters_refined'] = 1

            self.estimation_options['ttest_selection_norefined'] = False
            self.estimation_options['link_selection'] = False

            self.set_known_pathset_mode(True)

            # Assumed mean to compute the benchmark mean model (default is None)
            self.estimation_results['mean_count_benchmark_model'] = mean_count

            self.estimation_options['k_Y'] = ['tt']
            self.estimation_options['k_Z'] = [attr for attr, value in theta.items() if attr != 'tt']

            self.theta_0 = {**{'tt': theta['tt']} , **{attr: value for attr, value in theta.items() if attr != 'tt'}}
        # self.theta_0 = dict.fromkeys(self.estimation_options['k_Y'] + self.estimation_options['k_Z'], 0)


    def set_simulated_counts(self, sd_x, max_link_coverage, snr_x = None, sd_Q = 0, scale_Q = 1, simulate: bool = True):

        # max link coverage because it may happen that the generated paths covered less than the maximum level of coverage required

        if simulate is True:
            print('\nEnabling simulation mode')

            # - Coverage of link observations (proportion)
            self.sim_options['max_link_coverage'] = max_link_coverage
            # sim_options['max_link_coverage'] = 0.7 #0.7

            # - Mean and standard deviation of noise for count measurements
            self.sim_options['noise_params'].update({'mu_x': 0, 'sd_x': sd_x, 'snr_x': snr_x})
            self.sim_options['noise_params'].update({'sd_Q': sd_Q, 'scale_Q': scale_Q})

            # Notes: # sd is defined as proportion of the mean of the count vector
            # 0.1 means a sd igual to the 10% of the counts mean
            # A 0.2 deviatino respect to the mean does not affect inference and ensure convergence

            self.sim_options['simulated_counts'] = True

        else:
            print('\nEstimation using real-world data')

            self.sim_options['simulated_counts'] = False






#TODO: Log files
# =============================================================================
# 4) LOG FILES AND EXCEPTIONS
# =============================================================================