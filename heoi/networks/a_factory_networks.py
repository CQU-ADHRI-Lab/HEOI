class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):
        if network_name == 'Dense_fusion_V1':
            from networks.network_DenseInter_pathfusion_v1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)
        return network
