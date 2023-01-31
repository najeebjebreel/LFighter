from environment_federated import *


def run_exp(dataset_name, model_name, dd_type,
    num_peers, frac_peers, seed, test_batch_size, criterion, global_rounds, 
    local_epochs, local_bs, local_lr , local_momentum , labels_dict, device, 
    attackers_ratio, attack_type, malicious_behavior_rate, rule, 
    class_per_peer, samples_per_class, rate_unbalance, alpha, source_class, target_class, resume):
    print('\n--> Starting experiment...')
    flEnv = FL(dataset_name = dataset_name, model_name = model_name, dd_type = dd_type, num_peers = num_peers, 
    frac_peers = frac_peers, seed = seed, test_batch_size = test_batch_size, criterion = criterion, global_rounds = global_rounds, 
    local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum, 
    labels_dict = labels_dict, device = device, attackers_ratio = attackers_ratio,
    class_per_peer = class_per_peer, samples_per_class = samples_per_class, 
    rate_unbalance = rate_unbalance, alpha = alpha, source_class = source_class)
    print('Data set:', dataset_name)
    print('Data distribution:', dd_type)
    print('Aggregation rule:', rule)
    print('Attack Type:', attack_type)
    print('Attackers Ratio:', np.round(attackers_ratio*100, 2), '%')
    print('Malicious Behavior Rate:', malicious_behavior_rate*100, '%')
    # flEnv.simulate(attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate,
    #                 from_class = from_class, to_class = to_class,
    #                  rule=rule)
    flEnv.run_experiment(attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                    source_class = source_class, target_class = target_class, 
                    rule=rule, resume = resume)
    print('\n--> End of Experiment.')
