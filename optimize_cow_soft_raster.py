import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def load_obj(path):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def normalize_verts(verts):
    verts = verts - verts.mean(0, keepdim=True)
    scale = verts.abs().max().clamp_min(1e-6)
    return verts / scale


def make_uv_sphere(rings=22, segments=44):
    verts = [[0.0, 1.0, 0.0]]
    for r in range(1, rings):
        theta = math.pi * r / rings
        for s in range(segments):
            phi = 2.0 * math.pi * s / segments
            verts.append([math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)])
    verts.append([0.0, -1.0, 0.0])

    faces = []
    south = len(verts) - 1
    for s in range(segments):
        faces.append([0, 1 + (s + 1) % segments, 1 + s])
    for r in range(rings - 2):
        row = 1 + r * segments
        nxt = row + segments
        for s in range(segments):
            a, b = row + s, row + (s + 1) % segments
            c, d = nxt + s, nxt + (s + 1) % segments
            faces.append([a, d, b])
            faces.append([a, c, d])
    last = 1 + (rings - 2) * segments
    for s in range(segments):
        faces.append([south, last + s, last + (s + 1) % segments])
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def camera_matrices(num_views, elevation_deg=12.0, device="cpu"):
    mats = []
    elev = math.radians(elevation_deg)
    rx = torch.tensor(
        [[1, 0, 0], [0, math.cos(elev), -math.sin(elev)], [0, math.sin(elev), math.cos(elev)]],
        dtype=torch.float32,
        device=device,
    )
    for i in range(num_views):
        az = 2.0 * math.pi * i / num_views
        ry = torch.tensor(
            [[math.cos(az), 0, math.sin(az)], [0, 1, 0], [-math.sin(az), 0, math.cos(az)]],
            dtype=torch.float32,
            device=device,
        )
        mats.append(rx @ ry)
    return torch.stack(mats, dim=0)


def project_orthographic(verts, cameras, scale=1.35):
    view = torch.einsum("bij,vj->bvi", cameras, verts)
    xy = view[..., :2] / scale
    return xy


def pixel_grid(image_size, device):
    y, x = torch.meshgrid(
        torch.linspace(1.0, -1.0, image_size, device=device),
        torch.linspace(-1.0, 1.0, image_size, device=device),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1).reshape(-1, 2)


def soft_silhouette(verts, faces, cameras, image_size=64, sigma=0.02, face_chunk=512):
    device = verts.device
    points = pixel_grid(image_size, device)
    proj = project_orthographic(verts, cameras)
    images = []
    for b in range(proj.shape[0]):
        alpha = torch.zeros(points.shape[0], device=device)
        tri_all = proj[b, faces]
        for start in range(0, faces.shape[0], face_chunk):
            tri = tri_all[start : start + face_chunk]
            v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
            area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
            valid = area.abs() > 1e-5
            sign = torch.where(area >= 0, 1.0, -1.0).view(-1, 1)

            def edge_distance(a, bb):
                edge = bb - a
                rel = points.unsqueeze(0) - a.unsqueeze(1)
                cross = edge[:, 0:1] * rel[..., 1] - edge[:, 1:2] * rel[..., 0]
                return sign * cross / edge.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            d = torch.minimum(torch.minimum(edge_distance(v0, v1), edge_distance(v1, v2)), edge_distance(v2, v0))
            prob = torch.sigmoid(d / sigma)
            prob = torch.where(valid.view(-1, 1), prob, torch.zeros_like(prob))
            # Max aggregation keeps the boundary differentiable without letting
            # thousands of far-away triangles accumulate into a white background.
            alpha = torch.maximum(alpha, prob.max(dim=0).values)
        alpha = ((alpha - 0.5) * 2.0).clamp(0.0, 1.0)
        images.append(alpha.reshape(image_size, image_size))
    return torch.stack(images, dim=0)


def mesh_edges(faces):
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def face_adjacency_pairs(faces):
    edge_to_faces = {}
    for fi, face in enumerate(faces.tolist()):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((a, b)))
            edge_to_faces.setdefault(key, []).append(fi)
    pairs = [fs[:2] for fs in edge_to_faces.values() if len(fs) >= 2]
    return torch.tensor(pairs, dtype=torch.long) if pairs else torch.empty((0, 2), dtype=torch.long)


def regularization_losses(verts, faces, edges, initial_edge_lengths, adjacent_faces):
    src, dst = edges[:, 0], edges[:, 1]
    deg = torch.zeros(verts.shape[0], device=verts.device).index_add_(0, src, torch.ones_like(src, dtype=verts.dtype))
    deg = deg.index_add(0, dst, torch.ones_like(dst, dtype=verts.dtype)).clamp_min(1.0)
    nbr_sum = torch.zeros_like(verts)
    nbr_sum.index_add_(0, src, verts[dst])
    nbr_sum.index_add_(0, dst, verts[src])
    lap = ((verts - nbr_sum / deg[:, None]) ** 2).sum(dim=1).mean()

    lengths = (verts[src] - verts[dst]).norm(dim=1)
    edge = (((lengths - initial_edge_lengths) / initial_edge_lengths.clamp_min(1e-6)) ** 2).mean()

    if adjacent_faces.numel() == 0:
        normal = verts.new_tensor(0.0)
    else:
        tri = verts[faces]
        normals = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
        n0, n1 = normals[adjacent_faces[:, 0]], normals[adjacent_faces[:, 1]]
        normal = (1.0 - (n0 * n1).sum(dim=-1)).mean()
    return lap, edge, normal


def save_obj(path, verts, faces, colors=None):
    with open(path, "w", encoding="utf-8") as f:
        verts_cpu = verts.detach().cpu()
        colors_cpu = colors.detach().cpu().clamp(0, 1) if colors is not None else None
        for i, v in enumerate(verts_cpu.tolist()):
            if colors_cpu is None:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            else:
                c = colors_cpu[i].tolist()
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        for face in faces.detach().cpu().tolist():
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def procedural_cow_colors(verts):
    base = torch.full_like(verts, 0.88)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    spots = (
        torch.sin(8.0 * x + 2.5 * z)
        + torch.sin(10.0 * z - 1.5 * y)
        + torch.sin(7.0 * (x + y))
    )
    dark = (spots > 1.05).float().unsqueeze(1)
    color = base * (1.0 - dark) + torch.full_like(verts, 0.08) * dark
    muzzle = ((y < -0.15) & (z < -0.35)).float().unsqueeze(1)
    color = color * (1.0 - muzzle) + torch.tensor([0.82, 0.58, 0.52], device=verts.device) * muzzle
    return color.clamp(0.0, 1.0)


def soft_rgb_render(verts, faces, vertex_colors, cameras, image_size=64, sigma=0.02, face_chunk=512):
    device = verts.device
    points = pixel_grid(image_size, device)
    proj = project_orthographic(verts, cameras)
    images, silhouettes = [], []
    face_colors = vertex_colors[faces].mean(dim=1)
    for b in range(proj.shape[0]):
        weight_sum = torch.zeros(points.shape[0], device=device)
        rgb_sum = torch.zeros(points.shape[0], 3, device=device)
        tri_all = proj[b, faces]
        for start in range(0, faces.shape[0], face_chunk):
            tri = tri_all[start : start + face_chunk]
            colors = face_colors[start : start + face_chunk]
            v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
            area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
            valid = area.abs() > 1e-5
            sign = torch.where(area >= 0, 1.0, -1.0).view(-1, 1)

            def edge_distance(a, bb):
                edge = bb - a
                rel = points.unsqueeze(0) - a.unsqueeze(1)
                cross = edge[:, 0:1] * rel[..., 1] - edge[:, 1:2] * rel[..., 0]
                return sign * cross / edge.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            d = torch.minimum(torch.minimum(edge_distance(v0, v1), edge_distance(v1, v2)), edge_distance(v2, v0))
            weight = ((torch.sigmoid(d / sigma) - 0.5) * 2.0).clamp(0.0, 1.0)
            weight = torch.where(valid.view(-1, 1), weight, torch.zeros_like(weight))
            rgb_sum = rgb_sum + weight.transpose(0, 1) @ colors
            weight_sum = weight_sum + weight.sum(dim=0)
        alpha = weight_sum.clamp(0.0, 1.0)
        rgb = rgb_sum / weight_sum.clamp_min(1e-6).unsqueeze(1)
        rgb = rgb * alpha.unsqueeze(1)
        images.append(rgb.reshape(image_size, image_size, 3))
        silhouettes.append(alpha.reshape(image_size, image_size))
    return torch.stack(images, dim=0), torch.stack(silhouettes, dim=0)


def save_progress(path, target, current, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, image, name in zip(axes, [target, current], ["Ground Truth Silhouette", title]):
        image = image.detach().cpu()
        image = (image > 0.02).float()
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rgb_progress(path, target_rgb, pred_rgb, target_sil, pred_sil, title):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    target_rgb = target_rgb.detach().cpu()
    pred_rgb = pred_rgb.detach().cpu()
    target_rgb = (target_rgb / target_rgb.max().clamp_min(1e-6)).clamp(0, 1)
    pred_rgb = (pred_rgb / pred_rgb.max().clamp_min(1e-6)).clamp(0, 1)
    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Target RGB")
    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title(title)
    axes[1, 0].imshow((target_sil.detach().cpu() > 0.02).float(), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Target Silhouette")
    axes[1, 1].imshow((pred_sil.detach().cpu() > 0.02).float(), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted Silhouette")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Differentiable soft-rasterization mesh fitting demo.")
    parser.add_argument("--target", default="work/cow.obj")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--w-lap", type=float, default=0.08)
    parser.add_argument("--w-edge", type=float, default=0.25)
    parser.add_argument("--w-normal", type=float, default=0.01)
    parser.add_argument("--fit-rgb", action="store_true", help="also optimize per-vertex RGB colors")
    parser.add_argument("--w-rgb", type=float, default=1.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    target_verts, target_faces = load_obj(args.target)
    target_verts = normalize_verts(target_verts).to(device)
    target_faces = target_faces.to(device)

    source_verts, source_faces = make_uv_sphere()
    source_verts = (0.72 * source_verts).to(device)
    source_faces = source_faces.to(device)
    deform = torch.zeros_like(source_verts, requires_grad=True)
    color_logits = torch.zeros_like(source_verts, requires_grad=True) if args.fit_rgb else None

    cameras = camera_matrices(args.views, device=device)
    with torch.no_grad():
        target_sil = soft_silhouette(target_verts, target_faces, cameras, args.image_size, args.sigma, face_chunk=384)
        target_sil = (target_sil > 0.02).float()
        if args.fit_rgb:
            target_colors = procedural_cow_colors(target_verts)
            target_rgb, target_rgb_sil = soft_rgb_render(
                target_verts, target_faces, target_colors, cameras, args.image_size, args.sigma, face_chunk=384
            )
            target_rgb = target_rgb.detach()
            target_rgb_sil = (target_rgb_sil > 0.02).float()

    edges = mesh_edges(source_faces).to(device)
    adjacent_faces = face_adjacency_pairs(source_faces.cpu()).to(device)
    initial_edge_lengths = (source_verts[edges[:, 0]] - source_verts[edges[:, 1]]).norm(dim=1).detach()
    params = [deform] if color_logits is None else [deform, color_logits]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    losses = []

    for it in range(args.iters):
        optimizer.zero_grad()
        verts = source_verts + deform
        if args.fit_rgb:
            pred_rgb, pred_sil = soft_rgb_render(verts, source_faces, torch.sigmoid(color_logits), cameras, args.image_size, args.sigma)
            rgb_loss = F.mse_loss(pred_rgb, target_rgb)
            sil_target = target_rgb_sil
        else:
            pred_sil = soft_silhouette(verts, source_faces, cameras, args.image_size, args.sigma)
            rgb_loss = verts.new_tensor(0.0)
            sil_target = target_sil
        sil_loss = F.mse_loss(pred_sil, sil_target)
        lap_loss, edge_loss, normal_loss = regularization_losses(verts, source_faces, edges, initial_edge_lengths, adjacent_faces)
        loss = args.w_rgb * rgb_loss + sil_loss + args.w_lap * lap_loss + args.w_edge * edge_loss + args.w_normal * normal_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            deform.clamp_(-0.75, 0.75)
        losses.append([loss.item(), sil_loss.item(), rgb_loss.item(), lap_loss.item(), edge_loss.item(), normal_loss.item()])
        if it % 10 == 0 or it == args.iters - 1:
            print(
                f"iter {it:04d}/{args.iters} total={loss.item():.5f} sil={sil_loss.item():.5f} "
                f"rgb={rgb_loss.item():.5f} lap={lap_loss.item():.5f} edge={edge_loss.item():.5f} normal={normal_loss.item():.5f}",
                flush=True,
            )

    final_verts = source_verts + deform.detach()
    final_colors = torch.sigmoid(color_logits.detach()) if args.fit_rgb else None
    if args.fit_rgb:
        final_rgb, final_sil = soft_rgb_render(final_verts, source_faces, final_colors, cameras, args.image_size, args.sigma)
        save_obj(outdir / "optimized_textured_cow_like_mesh.obj", final_verts, source_faces, final_colors)
        save_rgb_progress(
            outdir / "rgb_texture_fit.png",
            target_rgb[0],
            final_rgb[0],
            target_rgb_sil[0],
            final_sil[0],
            f"Optimized RGB ({args.iters} iters)",
        )
    else:
        final_sil = soft_silhouette(final_verts, source_faces, cameras, args.image_size, args.sigma)
    save_obj(outdir / "optimized_cow_like_mesh.obj", final_verts, source_faces, final_colors)
    save_progress(outdir / "silhouette_fit.png", target_sil[0], final_sil[0], f"Optimized ({args.iters} iters)")

    loss_tensor = torch.tensor(losses)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_tensor[:, 0], label="total")
    ax.plot(loss_tensor[:, 1], label="silhouette")
    if args.fit_rgb:
        ax.plot(loss_tensor[:, 2], label="rgb")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close(fig)

    print(f"saved: {outdir / 'optimized_cow_like_mesh.obj'}")
    print(f"saved: {outdir / 'silhouette_fit.png'}")
    print(f"saved: {outdir / 'loss_curve.png'}")
    if args.fit_rgb:
        print(f"saved: {outdir / 'optimized_textured_cow_like_mesh.obj'}")
        print(f"saved: {outdir / 'rgb_texture_fit.png'}")


if __name__ == "__main__":
    main()
