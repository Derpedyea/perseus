#!/usr/bin/env python3
"""Splits an op list from llama.cpp's test-export-graph-ops for the lab's GPU self-test (step 3-02).

test-backend-ops --test-file replays each exported op with random inputs and compares the GPU's
output with the CPU's by normalized mean squared error. That comparison can't judge some ops, so
they are left out of the replay; OUT.builtin names them, and step 3-02 runs llama.cpp's own cases
for exactly those ops (and weight types) instead, so every op type a model uses is checked:
  index outputs     ARGSORT, TOP_K: the op guarantees an order or a set; tied random values
                    come back in another order (built-in cases use distinct values / compare sets)
  outside domain    SQRT: random negative inputs give NaN on the GPU and the CPU alike
  reductions with   MUL_MAT, SUM_ROWS with under 64 output values: the error is then relative to
  tiny outputs      a few sums of random terms, near zero at random (seen: a bf16 2560 -> 1 dot
                    product off by 7.5e-4 against a 5e-4 limit in 1 of 12 replays)
  routed experts    MUL_MAT_ID: they stay in RAM (CPU), and quantizing a whole expert tensor of
                    random weights takes minutes per case on the lab CPU

usage: gpu_ops.py GGML_H IN OUT   (GGML_H gives the op and type numbering the exported file uses)
writes OUT (the replay list) and OUT.builtin ("OP<TAB>-p filter or -" per line)
"""
import re
import sys

INDEX_OUTPUT = {"ARGSORT", "TOP_K"}
OUTSIDE_DOMAIN = {"SQRT"}
REDUCTIONS = {"MUL_MAT", "SUM_ROWS"}
TINY = 64
ALWAYS_BUILTIN = INDEX_OUTPUT | OUTSIDE_DOMAIN | {"MUL_MAT_ID"}
# built-in cases of these ops name their weight type type_a (test_mul_mat, test_mul_mat_id)
TYPE_PARAM = {"MUL_MAT": "type_a", "MUL_MAT_ID": "type_a"}


def enums(ggml_h):
    src = open(ggml_h).read()
    body = re.search(r"enum ggml_op \{(.*?)\};", src, re.S).group(1)
    ops = [n[len("GGML_OP_"):] for n in re.findall(r"^\s*(GGML_OP_[A-Z0-9_]+)", body, re.M)]
    if ops[0] != "NONE" or ops[-1] != "COUNT" or not (ALWAYS_BUILTIN | REDUCTIONS) <= set(ops):
        sys.exit(f"unexpected enum ggml_op in {ggml_h}")
    body = re.search(r"enum ggml_type \{(.*?)\};", src, re.S).group(1)
    types = {int(v): n for n, v in re.findall(r"GGML_TYPE_([A-Z0-9_]+)\s*=\s*(\d+)", body)}
    if types.get(0) != "F32":
        sys.exit(f"unexpected enum ggml_type in {ggml_h}")
    return ops, types


def type_name(enum_name):
    """GGML_TYPE_Q5_K -> q5_K, as ggml_type_name() spells it in test-backend-ops parameters."""
    n = enum_name.lower()
    return n[:-2] + "_K" if n.endswith("_k") else n


def main():
    ggml_h, src, dst = sys.argv[1:4]
    ops, types = enums(ggml_h)
    kept, builtin = [], {}
    for line in open(src):
        f = line.split()
        if not f:
            continue
        # op type ne0..ne3 n_params params... n_src [src_type ne0..3 nb0..3]... name
        op = ops[int(f[0])]
        n_out = int(f[2]) * int(f[3]) * int(f[4]) * int(f[5])
        i = 7 + int(f[6])
        src0 = type_name(types[int(f[i + 1])]) if int(f[i]) > 0 else None
        if op in ALWAYS_BUILTIN or (op in REDUCTIONS and n_out < TINY):
            builtin.setdefault(op, set())
            if op in TYPE_PARAM and src0:
                builtin[op].add(src0)
        else:
            kept.append(line)
    if not kept:
        sys.exit(f"{src}: no ops left to replay")
    with open(dst, "w") as out:
        out.writelines(kept)
    with open(dst + ".builtin", "w") as out:
        for op in sorted(builtin):
            t = sorted(builtin[op])
            out.write(f"{op}\t{TYPE_PARAM[op]}=({'|'.join(t)})\n" if t else f"{op}\t-\n")
    print(f"{dst}: replays {len(kept)} ops; built-in cases for {', '.join(sorted(builtin)) or 'none'}")


main()
