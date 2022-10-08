import time

def fmt_time(secs):
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    res=[]
    if(h):
        res.append("{:.0f}h".format(h))
    if(m):
        res.append("{:.0f}m".format(m))
    res.append("{:.02f}s".format(s))
    return " ".join(res)

def status(t0, count, total):
    t1 = time.time() - t0
    exp_t = (t1 / (count)) * total - t1
    return f"Processed:{count}/{total} ({round(100 * count / total, 2)}%)" + f". Time Taken: {fmt_time(t1)}" + f". Expected Completion: {fmt_time(exp_t)}"

