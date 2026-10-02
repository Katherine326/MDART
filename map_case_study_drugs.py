# -*- coding: utf-8 -*-
"""
案例研究结果 CSV 药物名映射/修复脚本（只依赖 pandas，无需 torch/dgl 环境）

背景：case_study_Alzheimer.py 旧版 load_drug_id_and_name 误把
DrugFingerprint.csv 的“首列”（实为行号 0,1,2,…）当作 DrugBank ID，
导致落盘 CSV 中 DrugBank_ID 与 drug_name 两列都是纯数字。
本脚本对任意一份已生成的案例研究 CSV（含 drug_index 列）做离线重映射，
从 DrugInformation.csv（行序 = 药物索引序）补出真实 DB ID 与通用名。

用法（在 zuizhong 目录内）：
  python map_case_study_drugs.py --csv Result/CaseStudy/AD_top10_C-dataset_s1234.csv \
                                 --dataset C-dataset
  # 默认输出到 <同目录>/<原名去扩展名>_mapped.csv
"""

import os
import argparse

import pandas as pd


def load_drug_id_and_name(data_dir, n_drugs):
    """drug 索引 i -> (DrugBank ID, 通用名)。与 case_study_Alzheimer.py 修复后一致。

    1) DrugInformation.csv 行序即药物索引序（B: name,id,smiles / C: id,name,smiles /
       F: ,id,name,smiles），'id' 与 'name' 同文件同序取用；
    2) 校验失败退回 AllNode/Allnode.csv 中含 'DB' 前缀的 ID 列；
    3) 兜底纯数字索引。
    """
    n_drugs = int(n_drugs)

    def _ids_look_like_db(series, need=n_drugs):
        vals = [str(v).strip() for v in series.tolist()
                if str(v).strip().lower() not in ('nan', '')]
        return vals[:need], (len(vals) >= need and vals[0].upper().startswith('DB'))

    db_ids, id2name = [], {}

    try:
        di = pd.read_csv(data_dir + 'DrugInformation.csv')
        ids_s = di['id'].astype(str).str.strip()
        names_s = di['name'].astype(str).str.strip()
        vals, ok = _ids_look_like_db(ids_s)
        if ok:
            db_ids = vals
            id2name = dict(zip(ids_s.tolist()[:n_drugs], names_s.tolist()[:n_drugs]))
    except Exception:
        pass

    if len(db_ids) != n_drugs:
        for fn in ('AllNode.csv', 'Allnode.csv'):
            try:
                an = pd.read_csv(data_dir + fn)
                if an.shape[1] == 1:
                    an = pd.read_csv(data_dir + fn, header=None)
                for col in an.columns:
                    vals, ok = _ids_look_like_db(an[col])
                    if ok and len(vals) == n_drugs:
                        db_ids = vals
                        break
            except Exception:
                continue
            if len(db_ids) == n_drugs:
                break
    if len(db_ids) == n_drugs and not id2name:
        try:
            di = pd.read_csv(data_dir + 'DrugInformation.csv')
            id2name = dict(zip(di['id'].astype(str).str.strip(),
                               di['name'].astype(str).str.strip()))
        except Exception:
            pass

    if len(db_ids) != n_drugs:
        db_ids = [str(i) for i in range(n_drugs)]
    return [(did, id2name.get(did, did)) for did in db_ids]


def main():
    ap = argparse.ArgumentParser(description='案例研究 CSV 药物索引 -> 真实 DB ID/名称')
    ap.add_argument('--csv', required=True, help='待修复的案例研究结果 CSV')
    ap.add_argument('--dataset', default='C-dataset', help='F-dataset / C-dataset / B-dataset')
    ap.add_argument('--data_root', default='data', help='数据根目录（默认 data）')
    ap.add_argument('--out', default=None, help='输出路径（默认 <同名>_mapped.csv）')
    args = ap.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset) + os.sep
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    if 'drug_index' not in df.columns:
        raise SystemExit('输入 CSV 缺少 drug_index 列：%s' % (df.columns.tolist(),))

    # 药物总数 = DrugFingerprint.csv 数据行数（首行是表头）；失败则取 max(drug_index)+1
    try:
        n_drugs = len(pd.read_csv(data_dir + 'DrugFingerprint.csv'))
    except Exception:
        n_drugs = int(df['drug_index'].max()) + 1

    table = load_drug_id_and_name(data_dir, n_drugs)
    if n_drugs != len(table):
        raise SystemExit('药物索引表解析异常：n_drugs=%d 但仅解析到 %d 条' % (n_drugs, len(table)))

    rows, bad = [], []
    for _, r in df.iterrows():
        i = int(r['drug_index'])
        if i < 0 or i >= len(table):
            bad.append(i)
            did = name = str(i)
        else:
            did, name = table[i]
        nr = r.to_dict()
        nr['DrugBank_ID'] = did
        nr['drug_name'] = name
        rows.append(nr)
    if bad:
        print('警告：以下 drug_index 超出 [0,%d)，无法映射 -> %s' % (n_drugs - 1, bad))

    out_df = pd.DataFrame(rows)
    out = args.out or os.path.splitext(args.csv)[0] + '_mapped.csv'
    out_df.to_csv(out, index=False, float_format='%.6f')
    print('Saved ->', out)

    # 控制台对照表
    pcol = 'P(positive)' if 'P(positive)' in out_df.columns else \
        ([c for c in out_df.columns if c.lower().startswith('p')] or [None])[0]
    kcol = 'known_AD_in_dataset' if 'known_AD_in_dataset' in out_df.columns else None
    print('\n%-4s %-10s %-9s %-24s %-12s %s' % ('rank', 'drug_index', 'DrugBank_ID',
                                                'drug_name', (pcol or 'P'), (kcol or '')))
    for _, r in out_df.iterrows():
        print('%-4s %-10s %-9s %-24s %-12s %s' % (
            r.get('rank', ''), r.get('drug_index', ''), r['DrugBank_ID'], r['drug_name'],
            (r[pcol] if pcol else ''), (r[kcol] if kcol else '')))


if __name__ == '__main__':
    main()
