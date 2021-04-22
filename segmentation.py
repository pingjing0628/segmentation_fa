from Seg import SegmentationMetricHelper

fa_data = {
    "image_removed": "false",
    "model_name": "segmentation_fa",
    "model_type": "Segmentation",
    "model_version": "v0.0.0.1",
    "values": [
        {
            "GT": [
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[16, 20, 36, 20, 48, 50, 36, 80, 16, 80]]
                },
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[26, 30, 46, 30, 58, 60, 46, 90, 26, 90]]
                },
                {
                    "category_name": "person",
                    "iscrowd": 0,
                    "segmentation": [[106, 110, 126, 110, 138, 140, 126, 170, 106, 170]]
                },
                {
                    "category_name": "person",
                    "iscrowd": 0,
                    "segmentation": [[116, 120, 136, 120, 148, 150, 136, 180, 116, 180]]
                }
            ],
            "IFR": [
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[18, 22, 38, 22, 50, 52, 38, 82, 18, 82]]
                },
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[300, 300, 350, 300, 350, 350, 300, 350, 333, 320]]
                },
                {
                    "category_name": "person",
                    "iscrowd": 0,
                    "segmentation": [[116, 120, 136, 120, 148, 150, 136, 180, 116, 180]]
                },

            ]
        },
        {
            "GT": [
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[16, 20, 36, 20, 48, 50, 36, 80, 16, 80]]
                },
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[26, 30, 46, 30, 58, 60, 46, 90, 26, 90]]
                },
                {
                    "category_name": "person",
                    "iscrowd": 0,
                    "segmentation": [[106, 110, 126, 110, 138, 140, 126, 170, 106, 170]]
                },
                {
                    "category_name": "person",
                    "iscrowd": 0,
                    "segmentation": [[116, 120, 136, 120, 148, 150, 136, 180, 116, 180]]
                }
            ],
            "IFR": [
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[18, 22, 38, 22, 50, 52, 38, 82, 18, 82]]
                },
                {
                    "category_name": "car",
                    "iscrowd": 0,
                    "segmentation": [[26, 30, 46, 30, 58, 60, 46, 90, 26, 90]]
                },
            ]
        }
    ]
}
helper = SegmentationMetricHelper(fa_data)
fa = helper.getFA()
print("fa", fa)
cm = helper.getCM()
print("cm", cm)
# kpi = helper.getKPI()
