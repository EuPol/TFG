from sklearn import metrics

EXPERIMENT_NAME='prueba'

#Read json file from  /mnt/cesar/ECAI/d_bowser/experiments/COX_run_EVT_with_CSV_and_limit_2/0_cluster_metrics.json
import json
with open('./experiments/prueba/0_classifier_final.json') as json_file:
    cluster_metrics = json.load(json_file)

pred_total=[]
true_total=[]

for user in cluster_metrics:
    pred_labels=cluster_metrics[user]['pred_labels']
    true_labels=cluster_metrics[user]['true_labels']
    pred_total+=pred_labels
    true_total+=true_labels

print('┌───────────────────────────┐')
print('│      Cluster Metrics      │')
print('├───────────────────────────┤')
print('│  Homogenity:     {:0.4f}   │'.format(metrics.homogeneity_score(true_total, pred_total)))
print('│  Completeness:   {:0.4f}   │'.format(metrics.completeness_score(true_total, pred_total)))
print('│  V-measure:      {:0.4f}   │'.format(metrics.v_measure_score(true_total, pred_total)))
print('│  Mutual Info:    {:0.4f}   │'.format(metrics.normalized_mutual_info_score(true_total, pred_total)))
print('│  Rand Score:     {:0.4f}   │'.format(metrics.rand_score(true_total, pred_total)))
print('└───────────────────────────┘')
