import copy
import heapq
import cv2 as cv
import numpy as np
import pandas as pd


class SegmentationMetricHelper(object):
    __requiredKeys = ['model_type', 'model_name', 'model_version']
    result = []
    metric = {}
    KPI = 'mAP'
    cm = {}
    iou_threshold = 0.5

    def __init__(self, fa_metric):
        self.metric = {key: fa_metric[key] for key in self.__requiredKeys}
        if set(list(self.metric.keys())) != set(self.__requiredKeys):
            raise ValueError("Keys:{} are not fully contained in FA data.".format(self.__requiredKeys))

        fa_value = fa_metric.get("values")
        if fa_value is None:
            raise ValueError("'values' is not found in FA metric.")

        results_list = []
        for value in fa_value:
            results = {}
            class_wise_map = {}
            value['results'] = {}
            # 根據種類分群
            for gt in value['GT']:
                category = gt['category_name']
                if not category in class_wise_map:
                    class_wise_map[category] = {"GT": [], "IFR": []}
                class_wise_map[category]["GT"].append(gt)
            for ifr in value['IFR']:
                category = ifr['category_name']
                if not category in class_wise_map:
                    class_wise_map[category] = {"GT": [], "IFR": []}
                class_wise_map[category]["IFR"].append(ifr)

            for category in class_wise_map:
                tp_iou_sum = 0
                fp_iou_sum = 0
                fn = 0
                tp = 0
                fp = 0
                tp_iou_mean = 0
                fp_iou_mean = 0
                fp_list = []

                gt_used = [False] * len(class_wise_map[category]['GT'])
                ifr_used = [-1] * len(class_wise_map[category]['IFR'])
                h = []
                for GT_idx in range(len(class_wise_map[category]['GT'])):
                    gt_segmentation = self._spilt_list(class_wise_map[category]['GT'][GT_idx]['segmentation'])
                    class_wise_map[category]['GT'][GT_idx]['index'] = GT_idx
                    for IFR_idx in range(len(class_wise_map[category]['IFR'])):
                        ifr_segmentation = self._spilt_list(class_wise_map[category]['IFR'][IFR_idx]['segmentation'])

                        # counting IOU
                        iou = self._intersection_rate(gt_segmentation, ifr_segmentation)
                        # 將 iou 放入 tuple
                        heap_tuple = (-iou, GT_idx, IFR_idx)
                        # 丟入 heap
                        heapq.heappush(h, heap_tuple)
                # 若 heap 不為空
                while h:
                    max_category_iou = heapq.heappop(h)

                    # 找到最大的就認為是 TP, gt_used 為 True, ifr_used 為 gt index, 且 gt index 值不存在 ifr_used 中
                    if -max_category_iou[0] >= 0.5 and max_category_iou[1] not in ifr_used:
                        gt_used[max_category_iou[1]] = True
                        ifr_used[max_category_iou[2]] = max_category_iou[1]
                        tp_iou_sum += -max_category_iou[0]

                    # 若不是就 continue, 因為有可能是其他人的 tp, 非直接為 fp
                    else:
                        if max_category_iou[1] in ifr_used and ifr_used[max_category_iou[2]] == -1:
                            if fp_list != []:
                                for fp_l in fp_list:
                                    if max_category_iou[2] != fp_l[2]:
                                        fp_list.append(max_category_iou)
                                    else:
                                        continue
                            else:
                                fp_list.append(max_category_iou)
                        continue

                # 放置 index
                max_ifr_index = max(gt_used)
                for i in range(len(ifr_used)):
                    if ifr_used[i] == -1:
                        max_ifr_index += 1
                        class_wise_map[category]['IFR'][i]['index'] = max_ifr_index
                    else:
                        class_wise_map[category]['IFR'][i]['index'] = ifr_used[i]

                # 計算 tp, fp
                # 計算 -1 in ifr_used 作為 fp
                fp = ifr_used.count(-1)
                for fp_l in fp_list:
                    if ifr_used[fp_l[2]] == -1:
                        fp_iou_sum += -fp_l[0]

                # 計算 fp iou mean
                if fp != 0:
                    fp_iou_mean = fp_iou_sum / fp

                # 計算 不為 -1 in ifr_used 作為 tp
                tp = len(ifr_used) - fp

                # 計算 tp iou mean
                if tp != 0:
                    tp_iou_mean = tp_iou_sum / tp

                # 計算 fn
                if len(gt_used) > len(ifr_used):
                    fn = len(gt_used) - len(ifr_used)

                if results.get(category) is None:
                    results[category] = {}
                results[category].update({"tp": {"count": tp, "iou_mean": tp_iou_mean},
                                          "fp": {"count": fp, "iou_mean": fp_iou_mean},
                                          "fn": {"count": fn, "iou_mean": 0}})
            value['results'].update(results)
            results_list.append(results)

        self.fa = fa_metric

        cols = ["category", "tp", "fp", "fn"]
        df_rows = []
        for result in results_list:
            for k, v in result.items():
                df_rows.append({"category": k, "tp": v['tp']['count'], "fp": v['fp']['count'], "fn": v['fn']['count']})
        df = pd.DataFrame(df_rows, columns=cols)
        self.imgDF = df.groupby('category').sum()

    def _intersection_rate(self, s1, s2):
        area, _intersection = cv.intersectConvexConvex(np.array(s1), np.array(s2))

        return area / (cv.contourArea(np.array(s1)) + cv.contourArea(np.array(s2)) - area)

    def _spilt_list(self, segmentation):
        spilt = []
        n = 2
        for seg_list in segmentation:
            for i in range((len(seg_list) + n - 1) // n):
                spilt.append(seg_list[i * n:(i + 1) * n])

        return spilt

    def getCM(self):
        categoryAggDf = self.imgDF.groupby(by=['category']).sum().groupby(level=[0]).cumsum()
        # print(categoryAggDf)
        table_name = list(categoryAggDf.index)
        table_value = []
        for index, row in categoryAggDf.iterrows():
            table_value.append({"name": index,
                                "value": [[int(row["fp"]), 0], [int(row["tp"]), int(row["fn"])]]})
        self.cm.update({"table_name": table_name,
                        "table_type": "confusion_matrix",
                        "x-axis": ["P", "N"],
                        "y-axis": ["N", "P"],
                        "tables_value": table_value
                        })

        recall_content = {}
        precision_content = {}
        f1_score_content = {}
        precision = 0
        recall = 0
        f1_score = 0
        for k, v in self.imgDF.iterrows():
            tp = v['tp']
            fp = v['fp']
            fn = v['fn']
            if not v['tp'] == 0 and not v['fp'] == 0:
                precision = tp / (tp + fp)
            if not v['tp'] == 0 and v['fn'] == 0:
                recall = tp / (tp + fn)
            f1_base = precision + recall
            if f1_base != 0:
                f1_score = (2 * precision * recall) / (precision + recall)

            precision_content.update({k: precision})
            recall_content.update({k: recall})
            f1_score_content.update({k: f1_score})

        self.cm.update({"precision_recall": {"results": {
            "Precision": precision_content,
            "Recall": recall_content,
            "F1-Score": f1_score_content}}})

        return self.cm

    def getFA(self):
        return self.fa

    def getKPI(self):
        # Known issue: definition need to be discussed
        return self.KPI
