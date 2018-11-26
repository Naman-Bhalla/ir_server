from flask import Flask, jsonify
from utils.data_processing import extractLanguageFeatures, extractVisualFeatures, language_feature_process_dict
from utils.utils import read_json
import caffe
import numpy as np
app = Flask(__name__)
caffe.set_mode_gpu()
caffe.set_device(0)
test_h5 = '/home/ubuntu/LocalizingMoments/data/average_fc7.h5'

visual_feature = 'feature_process_context'
language_feature = 'recurrent_embedding'
max_iter = 30000
snapshot_interval = 30000
loc = True
snapshot_dir = '/home/ubuntu/LocalizingMoments/snapshots'

language_extractor_fcn = extractLanguageFeatures
visual_extractor_fcn = extractVisualFeatures

language_process = language_feature_process_dict[language_feature]
#language_processor = language_process()
data_orig = read_json('static/10_vid.json')
#Flow Things
flow_deploy_net = '/home/ubuntu/LocalizingMoments/prototxts/deploy_clip_retrieval_flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt'
flow_snapshot_tag = 'flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2'
flow_test_h5 = '/home/ubuntu/LocalizingMoments/data/average_global_flow.h5'
flow_snapshot = '%s/%s_iter_%%d.caffemodel' % (snapshot_dir, flow_snapshot_tag)
flow_net = caffe.Net(flow_deploy_net, flow_snapshot % 30000, caffe.TEST)
#RGB Things
rgb_deploy_net = '/home/ubuntu/LocalizingMoments/prototxts/deploy_clip_retrieval_rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt'
rgb_snapshot_tag = 'rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2'
rgb_test_h5 = '/home/ubuntu/LocalizingMoments/data/average_fc7.h5'
rgb_snapshot = '%s/%s_iter_%%d.caffemodel' % (snapshot_dir, rgb_snapshot_tag)
rgb_net = caffe.Net(rgb_deploy_net, rgb_snapshot % 30000, caffe.TEST)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/query/<model_type>/<user_query>')
def serve(model_type, user_query):
    response = {'results':[]}
    params = {'feature_process': visual_feature, 'loc_feature': loc, 'loss_type': 'triplet',
              'batch_size': 120, 'features': test_h5, 'oversample': False, 'sentence_length': 50,
              'query_key': 'query', 'cont_key': 'cont', 'feature_key_p': 'features_p',
              'feature_time_stamp_p': 'feature_time_stamp_p',
              'feature_time_stamp_n': 'feature_time_stampe_n'}
    if model_type == 'rgb':
        net = rgb_net
    else:
        net = flow_net
    for el in data_orig:
        el['description'] = user_query
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    params['vocab_dict'] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    params['num_glove_centroids'] = num_glove_centroids
    thread_result = {}

    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations

    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations

    all_scores = {}
    for iter in range(snapshot_interval, max_iter + 1, snapshot_interval):
        all_scores[iter] = {}

        # determine score for segments in each video
        for id, d in enumerate(data):
            vis_features, loc_features = visual_feature_extractor.get_data_test({'video': d['video']})
            lang_features, cont = textual_feature_extractor.get_data_test(d)

            net.blobs['image_data'].data[...] = vis_features.copy()
            net.blobs['loc_data'].data[...] = loc_features.copy()

            for i in range(vis_features.shape[0]):
                net.blobs['text_data'].data[:, i, :] = lang_features
                net.blobs['cont_data'].data[:, i] = cont

            top_name = 'rank_score'
            net.forward()
            sorted_segments = [possible_segments[i] for i in np.argsort(net.blobs[top_name].data.squeeze())]
            all_scores[iter][d['annotation_id']] = net.blobs[top_name].data.squeeze().copy()
            response['results'].append({
                'video': d['dl_link'],
                'segments': sorted_segments[:5]
            })
            # if id % 10 == 0:
            #     sys.stdout.write('\r%d/%d' % (id, len(data)))
    return jsonify(response)
    #     eval_predictions(sorted_segments_list, data)
    #
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    #
    # pkl.dump(all_scores, open('%s/%s_%s.p' % (result_dir, snapshot_tag, split), 'w'))
    # print("Dumped results to: %s/%s_%s.p" % (result_dir, snapshot_tag, split))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
