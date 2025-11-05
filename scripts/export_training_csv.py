import argparse
import csv
import mysql.connector


def main():
    parser = argparse.ArgumentParser(description='Export training CSV with object/behavior/proximity risks and label_level')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--user', default='root')
    parser.add_argument('--password', default='')
    parser.add_argument('--database', default='model_evaluation')
    parser.add_argument('--experiment', default=None, help='Filter by experiment_id')
    parser.add_argument('--outfile', required=True, help='Output CSV path')
    args = parser.parse_args()

    conn = mysql.connector.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
    )
    cur = conn.cursor(dictionary=True)
    sql = (
        "SELECT experiment_id, frame_id, object_risk, behavior_risk, proximity_risk, predicted_level "
        "FROM experiment_log"
    )
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
        writer.writerow([
            'experiment_id', 'frame_id',
            'object_risk', 'behavior_risk', 'proximity_risk',
            'label_level'
        ])
        for r in rows:
            writer.writerow([
                r.get('experiment_id'), r.get('frame_id'),
                r.get('object_risk'), r.get('behavior_risk'), r.get('proximity_risk'),
                r.get('predicted_level')
            ])

    print(f"Exported {len(rows)} rows -> {args.outfile}")


if __name__ == '__main__':
    main()


