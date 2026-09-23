"""Conservative date-level FIFO compatibility, not proven money provenance."""
from collections import defaultdict, deque

import pandas as pd


def temporal_features(nodes, tx, end, window=2):
    end = pd.Timestamp(end)
    cutoff = end - pd.Timedelta(days=window)
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for row in tx.itertuples(index=False):
        incoming[row.dst].append((row.date, round(row.sum_kzt * 100), row.src))
        outgoing[row.src].append((row.date, round(row.sum_kzt * 100), row.dst))
    result = []
    for gid in nodes.gid:
        ins, outs = incoming[gid], outgoing[gid]
        days = sorted({x[0] for x in ins + outs})
        by_in, by_out, payers = defaultdict(int), defaultdict(int), defaultdict(set)
        for date, amount, src in ins:
            by_in[date] += amount
            payers[date].add(src)
        for date, amount, _ in outs:
            by_out[date] += amount
        queue = deque()
        matched = 0
        eligible = sum(amount for date, amount, _ in ins if date <= cutoff)
        for day in days:
            while queue and (day - queue[0][0]).days > window:
                queue.popleft()
            # Outgoing processed first: unknown same-day ordering never becomes evidence.
            left = by_out[day]
            while left and queue:
                date, amount = queue[0]
                take = min(left, amount)
                if date <= cutoff:
                    matched += take
                left -= take
                if take == amount:
                    queue.popleft()
                else:
                    queue[0] = (date, amount - take)
            if by_in[day]:
                queue.append((day, by_in[day]))
        daily_total = [by_in[d] + by_out[d] for d in days]
        result.append({
            "gid": int(gid),
            "fast_1_2d_share": matched / eligible if eligible else 0.0,
            "temporal_eligible_kzt": eligible / 100,
            "temporal_matched_kzt": matched / 100,
            "max_same_day_payers": max(map(len, payers.values()), default=0),
            "active_days": len(days),
            "peak_day_share": max(daily_total, default=0) / max(1, sum(daily_total)),
            "late_inflow": any(date > cutoff for date, _, _ in ins),
            "first_seen": min(days).strftime("%Y-%m-%d") if days else "",
            "last_seen": max(days).strftime("%Y-%m-%d") if days else "",
        })
    return pd.DataFrame(result)
