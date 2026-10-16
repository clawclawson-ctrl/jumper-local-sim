"""CASTLE theme (rgba / lights only; physics never reads them) + perception colour families incl. GREEN. SIM-ONLY."""
from __future__ import annotations
import re
from jhs.soccer import brain as SBR

SBR.FAM.setdefault("green", dict(h=((95, 150),), s=0.45, v=(0.15, 1.0)))
FAMS = {"A": "red", "B": "blue", "C": "green"}


def theme(model):
    import mujoco
    for t in range(model.ntex):          # dusk sky
        if int(model.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            a = model.tex_adr[t]; nb = model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
            px = model.tex_data[a:a + nb].reshape(-1, model.tex_nchannel[t]); px[:, :3] = (px[:, :3].astype(float) * [0.25, 0.18, 0.14]).astype(px.dtype)
    model.vis.headlight.ambient[:] = [0.42, 0.38, 0.33]; model.vis.headlight.diffuse[:] = [0.45, 0.40, 0.34]
    n = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name == "ground" or name.startswith("wl_"): model.geom_matid[g] = -1; n += 1
    return n
