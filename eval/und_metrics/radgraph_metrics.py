import os
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from radgraph import F1RadGraph
from radgraph.radgraph import CACHE_DIR

class F1RadGraphv2(F1RadGraph):
    def __init__(self, reward_level, **kwargs):
        self._download_radgraph()
        super().__init__(reward_level, **kwargs)
        assert reward_level in ["simple", "partial", "complete", "all"]

    def _download_radgraph(self):
        import shutil
        local_src = os.environ.get("RADGRAPH_PATH")
        local_dst = os.path.join(CACHE_DIR, "radgraph.tar.gz")
        
        if not os.path.exists(local_dst):
            os.makedirs(CACHE_DIR, exist_ok=True)
            if os.path.exists(local_src):
                print(f"Copying RadGraph from {local_src} to {local_dst}")
                shutil.copy(local_src, local_dst)
            else:
                print(f"Downloading RadGraph to {local_dst}")
                hf_hub_download(
                    repo_id="StanfordAIMI/RRG_scorers",
                    filename="radgraph.tar.gz",
                    revision="d97745aa136e5beb927da7e768e99de6ae807902",
                    local_dir=CACHE_DIR,
                )

    def forward(self, refs, hyps):
        assert isinstance(hyps, str) or isinstance(hyps, list)
        assert isinstance(refs, str) or isinstance(refs, list)

        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(refs, str):
            refs = [refs]

        assert len(refs) == len(hyps)

        number_of_reports = len(hyps)
        empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        inference_dict = self.radgraph(report_list)

        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        
        for report_index in tqdm(range(number_of_reports), desc="Computing RadGraph rewards"):
            if report_index in empty_report_index_list:
                reward_list.append((0., 0., 0.))
                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports
        
        if self.reward_level == "all":
            reward_list_by_type = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            mean_reward = (np.mean(reward_list_by_type[0]), np.mean(reward_list_by_type[1]), np.mean(reward_list_by_type[2]))
        else:
            mean_reward = np.mean(reward_list, axis=0)
            mean_reward = {'f1-radgraph': mean_reward[0], 'precision-radgraph': mean_reward[1], 'recall-radgraph': mean_reward[2]}

        return mean_reward

def compute_reward(hypothesis_annotation_list, reference_annotation_list, reward_level):
    assert reward_level in ["simple", "partial", "complete", "all"]
    if (len(hypothesis_annotation_list["entities"].keys()) == 0 or 
            len(reference_annotation_list["entities"].keys()) == 0):
        return (0., 0., 0.)
    
    simple = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)
    partial = exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list)
    complete = exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list)
    all = (simple, partial, complete)
    
    return eval(reward_level)

def exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            candidate.append((entity["tokens"], entity["label"]))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    epsilon = 1e-10
    precision = (
        sum([1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list)])
        / (len(hypothesis_relation_token_list) + epsilon)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum([1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list)])
        / (len(reference_relation_token_list) + epsilon)
        if len(reference_relation_token_list) > 0
        else 0.0
    )

    f1_score = (
        (2 * precision * recall / (precision + recall + epsilon))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall

def exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    epsilon = 1e-10
    precision = (
        sum([1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list)])
        / (len(hypothesis_relation_token_list) + epsilon)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum([1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list)])
        / (len(reference_relation_token_list) + epsilon)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall + epsilon))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall

def exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.extend([(entity["tokens"].lower(),
                                  entity["label"],
                                  r[0],
                                  annotation_list["entities"][r[1]]["tokens"].lower())
                                 for r in entity["relations"]])

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    
    epsilon = 1e-10
    precision = (
        sum([1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list)])
        / (len(hypothesis_relation_token_list) + epsilon)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum([1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list)])
        / (len(reference_relation_token_list) + epsilon)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall + epsilon))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall

def calculate_radgraph_metrics(predictions, references, radgraph_level="partial", model_type="radgraph"):
    """Calculate RadGraph related metrics"""
    try:
        radgraph_evaluator = F1RadGraphv2(reward_level=radgraph_level, model_type=model_type)
        radgraph_scores = radgraph_evaluator(refs=references, hyps=predictions)

        # Format results as dictionary
        if isinstance(radgraph_scores, dict):
            return radgraph_scores
        else:
            # If tuple is returned (all mode)
            return {
                'RadGraph_Simple': radgraph_scores[0],
                'RadGraph_Partial': radgraph_scores[1],
                'RadGraph_Complete': radgraph_scores[2]
            }
    except Exception as e:
        print(f"Error computing RadGraph metrics: {e}")
        return {} 