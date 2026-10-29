"""Registered training-side measurements over the FULL 120 optimizer steps.

  A3.2 / A5.2  exposure, dose, survival per cohort per arm (dose source replaced
               per A5.2: std_reward>0 in prompt_reward_stats.csv, which IS the
               filter criterion; the clipfrac file was withdrawn).
  A4.3 / A7.4  two-sided waste, TWO measures with TWO verdicts and no silent
               rescue: (i) discard counts split all-wrong/all-correct, governed
               by the A4.3 drop rule; (ii) advantage mass, reported separately.
"""
import glob, json, os
import numpy as np, pandas as pd

EXP=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(EXP,"..","..","unlearnability-rlvr"))
BASE=os.path.join(REPO,"checkpoints","qwen_0.5b_baseline_1gpu_Qwen2.5-0.5B_simplelr_qwen_level1to4_sub1k_max_response1024_batch32_ppomini16_logprobbatch256_rollout8_klcoef0.0001_entcoef0.001_lr5e-7_valbatch200_seed1","prompt_reward_stats.csv")
G=8
sub=pd.read_csv(os.path.join(EXP,"adaprefix","eval_subset.csv")); rates=pd.read_csv(os.path.join(EXP,"adaprefix","armB_rho.csv"))
COH=[("D_u 71",set(sub[(sub.group=="D_u")&(sub.cap_zero==0)].qid)),
     ("control 74",set(sub[sub.group=="control"].qid)),
     ("confound-exposed 30",set(sub[(sub.group=="control")&(sub.confound_exposed==1)].qid)),
     ("full 1023",set(rates.qid))]
runs={"baseline":BASE}
for a in "BA":
    runs[f"Arm {a}"]=glob.glob(os.path.join(EXP,"smoke_ap","checkpoints",f"ap_arm{a}_u0.40_s999_*","prompt_reward_stats.csv"))[0]
D={k:pd.read_csv(v).query("step<=120") for k,v in runs.items()}
for k,v in D.items():
    v["surv"]=v.std_reward>0; v["k"]=(v.mean_reward*G).round().astype(int)
    v["allwrong"]=v.mean_reward<=0; v["allright"]=v.mean_reward>=1
    p=v.k/G; v["mass"]=np.where(v.surv, G*np.sqrt(p*(1-p)), 0.0)

out=["# Registered training-side measurements, full 120 optimizer steps\n",
"## A3.2 / A5.2 — exposure, dose and survival per cohort per arm\n",
"Dose = groups surviving `std > 0` (the filter criterion itself). The A3.2 source "
"`prompt_clipfrac_stats.csv` was withdrawn under its own validity rule: its "
"`prompt_id` is a within-batch index 0-31 with zero overlap with the 1023 qids.\n",
"| cohort | run | exposure | dose | dose/prompt | survival |","|---|---|---|---|---|---|"]
for nm,Q in COH:
    for r in ["baseline","Arm B","Arm A"]:
        x=D[r][D[r].prompt_id.isin(Q)]
        out.append(f"| {nm} | {r} | {len(x)} | {int(x.surv.sum())} | {x.surv.sum()/len(Q):.3f} | {x.surv.mean():.3f} |")
out.append("")
out.append("Structural check: surviving groups per step = " + ", ".join(
    f"{r} {D[r].groupby('step').surv.sum().mean():.1f}" for r in D) + " (train_batch_size 32).")

out+=["\n## A4.3 — discard counts, split by cause (MEASURE 1 of 2)\n",
"| cohort | run | groups | discarded | all-wrong | **all-correct** | wasted rollouts |","|---|---|---|---|---|---|---|"]
allcorrect={}
for nm,Q in COH:
    for r in ["baseline","Arm B","Arm A"]:
        x=D[r][D[r].prompt_id.isin(Q)]; disc=(~x.surv).sum()
        aw=x.allwrong.mean(); ac=x.allright.mean(); allcorrect[(nm,r)]=ac
        out.append(f"| {nm} | {r} | {len(x)} | {disc/len(x):.3f} | {aw:.3f} | **{ac:.4f}** | {disc*G:,} |")
out.append("")
acA=allcorrect[("full 1023","Arm A")]; acB=allcorrect[("full 1023","Arm B")]; acBase=allcorrect[("full 1023","baseline")]
verdict = ("**CLAIM DROPPED.** " if acA<0.02 and acB<0.02 else "**CLAIM SUPPORTED.** ")
out.append(f"### A4.3 drop rule, applied as registered\n")
out.append(f"All-correct discards over the full 1023: baseline {acBase:.4f}, Arm B {acB:.4f}, Arm A {acA:.4f}.")
out.append(verdict + ("The all-correct end of the two-sided waste claim does not replicate at scale; "
  "per A4.3 the discard-based claim is dropped, not softened. All-wrong dominates entirely."
  if acA<0.02 and acB<0.02 else "Arm A's all-correct discard rate is materially above Arm B's."))

out+=["\n## A7.4 — advantage mass (MEASURE 2 of 2, reported on its own terms)\n",
"| cohort | baseline | Arm B | Arm A | B vs base | A vs base |","|---|---|---|---|---|---|"]
for nm,Q in COH:
    m={r: D[r][D[r].prompt_id.isin(Q)].mass.sum() for r in ["baseline","Arm B","Arm A"]}
    out.append(f"| {nm} | {m['baseline']:.1f} | {m['Arm B']:.1f} | {m['Arm A']:.1f} | "
               f"{100*(m['Arm B']/m['baseline']-1):+.1f}% | {100*(m['Arm A']/m['baseline']-1):+.1f}% |")
out.append("")
out.append("No silent rescue: the mass measure surviving does not reinstate the discard claim above.")
open(os.path.join(EXP,"analysis","dose_and_waste.md"),"w").write("\n".join(out)+"\n")
print("\n".join(out))
