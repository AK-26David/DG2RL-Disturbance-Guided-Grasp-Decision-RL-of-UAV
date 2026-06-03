import torch

from models.graspnet import DG2GraspNet

from perception.preprocess import preprocess
from perception.grasp_extraction import get_top_k_grasps
from perception.objectness import compute_objectness

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DG2GraspNet().to(DEVICE)

checkpoint = torch.load(
    "models/dg2_graspnet_model.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

def module1_forward(rgb, depth):

    x = preprocess(rgb, depth)

    with torch.no_grad():

        q, cos, sin, w = model(
            x.unsqueeze(0).to(DEVICE)
        )

    grasps = get_top_k_grasps(
        q,
        cos,
        sin,
        w,
        k=10
    )

    objectness = compute_objectness(rgb)

    return {

        "q": q,
        "cos": cos,
        "sin": sin,
        "w": w,

        "grasps": grasps,

        "x": x,

        "objectness": objectness
    }