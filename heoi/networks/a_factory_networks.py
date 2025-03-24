class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):
        if network_name == 'encoder_decoder_lstm':
            pass
        elif network_name == 'Dense_V1':
            from networks.network_DenseInter_V1_0 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V1_1':
            from networks.network_DenseInter_V1_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V1_2':
            from networks.network_DenseInter_V1_2 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V1_3':
            from networks.network_DenseInter_V1_3 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V1_3_1':
            from networks.network_DenseInter_V1_3_1 import AttentionPrediction
        elif network_name == 'Dense_V1_3_1_one':
            from networks.network_DenseInter_V1_3_1_one import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V2_0':
            from networks.network_DenseInter_V2_0 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V3_0':
            from networks.network_DenseInter_V3_0 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V4_0':
            from networks.network_DenseInter_V4_0 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V5_0':
            from networks.network_DenseInter_V5_4_5 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V5_2':
            from networks.network_DenseInter_V5_2 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_V6':
            from networks.network_DenseInter_V6_7_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_skeleton_0':
            from networks.network_DenseInter_V6_7_0_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_bodypart_0':
            from networks.network_DenseInter_V6_7_0_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_box_0':
            from networks.network_DenseInter_V6_7_0_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_box':
            from networks.network_DenseInter_V6_7_3_3_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_skeleton':
            from networks.network_DenseInter_V6_7_3_3_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_bodypart':
            from networks.network_DenseInter_V6_7_3_3_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'DenseFusion_v2':
            from networks.network_DenseInter_pathfusion_v2_5 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_bodypart_6':
            from networks.network_DenseInter_V6_7_6_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_box_6':
            from networks.network_DenseInter_V6_7_6_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_skeleton_6':
            from networks.network_DenseInter_V6_7_6_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_bodypart_4':
            from networks.network_DenseInter_V6_7_4_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_box_4':
            from networks.network_DenseInter_V6_7_4_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_skeleton_4':
            from networks.network_DenseInter_V6_7_4_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_bodypart_5':
            from networks.network_DenseInter_V6_7_5_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == 'Dense_box_5':
            from networks.network_DenseInter_V6_7_5_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_skeleton_5":
            from networks.network_DenseInter_V6_7_5_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_bodypart_3_1":
            from networks.network_DenseInter_V6_7_3_1_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_box_3_1":
            from networks.network_DenseInter_V6_7_3_1_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_skeleton_3_1":
            from networks.network_DenseInter_V6_7_3_1_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_box_3_2":
            from networks.network_DenseInter_V6_7_3_2_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_skeleton_3_2":
            from networks.network_DenseInter_V6_7_3_2_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_bodypart_3_2":
            from networks.network_DenseInter_V6_7_3_2_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_bodypart_3_3":
            from networks.network_DenseInter_V6_7_3_3_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_box_3_3":
            from networks.network_DenseInter_V6_7_3_3_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_skeleton_3_3":
            from networks.network_DenseInter_V6_7_3_3_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_1_1":
            from networks.network_DenseInter_pathfusion_v1_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_1_2":
            from networks.network_DenseInter_pathfusion_v1_2 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1":
            from networks.network_DenseInter_pathfusion_v2_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3":
            from networks.network_DenseInter_pathfusion_v2_1_3 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_1":
            from networks.network_DenseInter_pathfusion_v2_1_3_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_2":
            from networks.network_DenseInter_pathfusion_v2_1_3_2 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_4":
            from networks.network_DenseInter_pathfusion_v2_1_3_4 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_3":
            from networks.network_DenseInter_pathfusion_v2_1_3_3 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_5":
            from networks.network_DenseInter_pathfusion_v2_1_3_5 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_6":
            from networks.network_DenseInter_pathfusion_v2_1_3_6 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_7":
            from networks.network_DenseInter_pathfusion_v2_1_3_7 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_3_8":
            from networks.network_DenseInter_pathfusion_v2_1_3_8 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_4":
            from networks.network_DenseInter_pathfusion_v2_1_4 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_5":
            from networks.network_DenseInter_pathfusion_v2_1_5 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_1_6":
            from networks.network_DenseInter_pathfusion_v2_1_6 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_2":
            from networks.network_DenseInter_pathfusion_v2_2 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_3_1":
            from networks.network_DenseInter_pathfusion_v2_3_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_4":
            from networks.network_DenseInter_pathfusion_v2_4 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_1_1_1":
            from networks.network_DenseInter_pathfusion_v1_1_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_3_1_1":
            from networks.network_DenseInter_pathfusion_v2_3_1_1 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_fusion_2_10":
            from networks.network_DenseInter_pathfusion_v2_10 import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_box_1_1":
            from networks.network_DenseInter_v1_1_1_box import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_skeleton_1_1":
            from networks.network_DenseInter_v1_1_1_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        elif network_name == "Dense_bodypart_1_1":
            from networks.network_DenseInter_v1_1_1_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_1_2":
            from networks.network_DenseInter_v1_2_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_2_1":
            from networks.network_DenseInter_v2_1_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_skeleton_1_2":
            from networks.network_DenseInter_v1_2_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_skeleton_2_1":
            from networks.network_DenseInter_v2_1_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_skeleton_2_2":
            from networks.network_DenseInter_v2_2_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_skeleton_2_3_1":
            from networks.network_DenseInter_v2_3_1_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_skeleton_2_4":
            from networks.network_DenseInter_v2_4_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name  == "Dense_box_skeleton_2_1_3_1":
            from networks.network_DenseInter_v2_1_3_1_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_skeleton_1_1_1":
            from networks.network_DenseInter_v1_1_1_box_skeleton import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_2_2":
            from networks.network_DenseInter_v2_1_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_2_3_1":
            from networks.network_DenseInter_v2_3_1_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_2_4":
            from networks.network_DenseInter_v2_4_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_1_1_1":
            from networks.network_DenseInter_v1_1_1_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_box_bodypart_2_1_3_1":
            from networks.network_DenseInter_v2_1_3_1_box_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_1_2":
            from networks.network_DenseInter_v1_2_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_2_1":
            from networks.network_DenseInter_v2_1_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_2_2":
            from networks.network_DenseInter_v2_2_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_2_3_1":
            from networks.network_DenseInter_v2_3_1_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_2_4":
            from networks.network_DenseInter_v2_4_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_1_1_1":
            from networks.network_DenseInter_v1_1_1_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_skeleton_bodypart_2_1_3_1":
            from networks.network_DenseInter_v2_1_3_1_skeleton_bodypart import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_scene":
            from networks.network_scene import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_human":
            from networks.network_human import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_object":
            from networks.network_object import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_scene_object":
            from networks.network_scene_object import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        
        elif network_name == "Dense_human_object":
            from networks.network_human_object import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
            
        elif network_name == "Dense_human_scene":
            from networks.network_human_scene import AttentionPrediction
            network = AttentionPrediction(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)
        return network
