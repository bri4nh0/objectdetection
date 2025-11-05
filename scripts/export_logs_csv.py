import argparse
import csv
import mysql.connector
from typing import Optional


def main():
    parser = argparse.ArgumentParser(description='Export experiment_log rows to CSV for calibration/evaluation')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--user', default='root')
    parser.add_argument('--password', default='')
    parser.add_argument('--database', default='model_evaluation')
    parser.add_argument('--experiment', default=None, help='Filter by experiment_id')
    parser.add_argument('--outfile', required=True, help='Output CSV path')
    parser.add_argument('--label-level', type=int, default=3, help='Level treated as positive label (default: 3=CRITICAL)')
    parser.add_argument('--score-col', default='fusion_score', help='Which score to export (fusion_score or trd_uq_score if present)')
    args = parser.parse_args()

    conn = mysql.connector.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
    )
    cur = conn.cursor(dictionary=True)
    sql = "SELECT experiment_id, frame_id, fusion_score, trd_uq_score, predicted_level FROM experiment_log"
    if args.experiment:
        sql += " WHERE experiment_id = %s"
        cur.execute(sql, (args.experiment,))
    else:
        cur.execute(sql)

    rows = list(cur.fetchall())
    cur.close()
    conn.close()

    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment_id', 'frame_id', 'score', 'label'])
        for r in rows:
            score = r.get(args.score_col)
            if score is None:
                score = r.get('fusion_score')
            level = int(r.get('predicted_level') or 0)
            label = 1 if level >= args.label_level else 0
            writer.writerow([r.get('experiment_id'), r.get('frame_id'), score, label])

    print(f"Exported {len(rows)} rows -> {args.outfile}")


if __name__ == '__main__':
    main()


