
import math
import numpy as np
import pygame

# ----------------------------------------------------------------- settings
W, H = 540, 960                    # window (9:16 like the video)
CX, ROOT_Y = W // 2, 790           # where the flower is rooted (screen coords)
STEM_H = 470
LIFE_SEC = 29.0
TILT = 0.36                        # camera looks slightly down on the bloom

PINK   = np.array([255,  80, 190], float)
CORE   = np.array([255, 228, 246], float)
VIOLET = np.array([205, 115, 255], float)
GREY   = np.array([120, 118, 135], float)
TEAL   = np.array([ 60, 255, 215], float)
GOLD   = np.array([255, 225, 160], float)
DUST   = np.array([150, 180, 255], float)

rng = np.random.default_rng(7)


# ------------------------------------------------------------------ helpers
def smooth(x, start, width=0.2):
    s = np.clip((x - start) / width, 0, 1)
    return s * s * (3 - 2 * s)

def ease(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3

def mix(a, b, k):
    k = np.asarray(k, float)
    return a + (b - a) * (k[..., None] if k.ndim else k)


# ----------------------------------------------- fixed random geometry (once)
RING = np.concatenate([[r] * (r * 6 + 5) for r in range(4)])       # 5,11,17,23 petals
P, K = len(RING), 170                                              # petals, particles each
PHI0 = np.concatenate([2 * np.pi * np.arange(r * 6 + 5) / (r * 6 + 5) + r * 0.55
                       for r in range(4)])
PL = 56 + RING * 17.0                                              # petal length
PW = PL * (0.27 - RING * 0.025)                                    # petal half-width
PU = rng.random((P, K)) ** 0.85                                    # along the petal
_m = rng.random((P, K))                                            # edge / vein / fill
PV = np.where(_m < 0.5, np.sign(rng.random((P, K)) - 0.5) * (1 - 0.05 * rng.random((P, K))),
     np.where(_m < 0.7, rng.normal(0, 0.04, (P, K)), rng.uniform(-1, 1, (P, K))))
PE = np.where(_m < 0.5, 1.0, np.where(_m < 0.7, 0.8, 0.45))      # edges glow brightest
FALL_AT = rng.uniform(0.74, 0.93, P)                               # when each petal lets go
SPIN = rng.uniform(-1.4, 1.4, P)
PHASE = rng.uniform(0, 6.28, P)

LEAVES = [(0.22, -1, 95, 0.55), (0.42, 1, 112, 0.60), (0.60, -1, 82, 0.70)]  # frac, side, len, angle
LN = 230
_lm = rng.random((3, LN))
LU = rng.random((3, LN)) ** 0.9
LV = np.where(_lm < 0.55, np.sign(rng.random((3, LN)) - 0.5),
     np.where(_lm < 0.7, 0.0, rng.uniform(-1, 1, (3, LN))))
SJ = rng.normal(0, 1.1, (3, 900, 2))                               # stem thickness jitter

BN = 700
BG = rng.normal(0, 1, (BN, 2))
BI = rng.random(BN) * 0.5 + 0.3

SN = 170                                                           # pollen sparks
SP_PH, SP_AN = rng.random(SN), rng.uniform(0, 2 * np.pi, SN)
SP_SP = rng.uniform(40, 150, SN)

RN = 260                                                           # orbit rings
RT = np.linspace(0, 2 * np.pi, RN, endpoint=False)

DN = 260                                                           # floating dust
DX, DY = rng.uniform(0, W, DN), rng.uniform(0, H, DN)
DS, DP = rng.uniform(4, 18, DN), rng.uniform(0, 6.28, DN)


# --------------------------------------------------- your three life functions
def stem_len(life):
    return ease(life / 0.4) * STEM_H

def stem(life, t, droop=0.0):
    h = stem_len(life)
    y = np.linspace(0, h, 900)
    x = 14 * np.sin(y / 80 + t)
    f = y / STEM_H                                   # bend over as the flower wilts
    return x + droop * 150 * f ** 2, y * (1 - 0.12 * droop * f)

def bloom(life, t, top, droop):
    """4 rings of petals. Returns screen x, y, brightness, colour-gradient and ring for every particle."""
    op = smooth(life, 0.42 + RING * 0.04) * 1.3                    # petals(ring*6+5, open*1.3)
    o = op[:, None]
    vis = np.clip(op / 1.3 * 5, 0, 1)[:, None]
    Lg = PL[:, None] * (0.25 + 0.75 * np.sqrt(op / 1.3)[:, None])  # petals grow as they open
    th0 = o * (0.30 + 0.20 * RING[:, None]) + droop * (0.5 + 0.1 * RING[:, None])
    c = 0.2 + 0.45 * o + droop * 1.1                               # tips curl outwards / down
    u = PU
    r = Lg * (np.cos(th0) - np.cos(th0 + c * u)) / c
    z = Lg * (np.sin(th0 + c * u) - np.sin(th0)) / c
    lat = PV * PW[:, None] * (Lg / PL[:, None]) * np.sin(np.pi * u ** 0.75)
    phi = PHI0[:, None]
    X = r * np.cos(phi) - lat * np.sin(phi)
    Z = r * np.sin(phi) + lat * np.cos(phi)
    a = t * 0.45                                                   # the slow spin
    Xr = X * math.cos(a) + Z * math.sin(a)
    Zr = -X * math.sin(a) + Z * math.cos(a)
    up = z * math.cos(TILT) - Zr * math.sin(TILT)
    shade = 0.72 + 0.28 * np.clip(Zr / 90, -1, 1)                  # front petals brighter
    sx, sy = Xr, -up                                               # screen offsets from the head
    tl = droop * 1.15                                              # head hangs down
    sx, sy = sx * math.cos(tl) - sy * math.sin(tl), sx * math.sin(tl) + sy * math.cos(tl)
    sx, sy = sx + top[0], sy + top[1]

    # falling petals (gravity = 9.8 * droop, like your wilt())
    tau = np.maximum(0, (life - FALL_AT) * LIFE_SEC)[:, None]
    if tau.max() > 0:
        g = 9.8 * droop
        mx, my = sx.mean(1, keepdims=True), sy.mean(1, keepdims=True)
        ang = SPIN[:, None] * np.minimum(tau, 2.2)
        dx, dy = sx - mx, sy - my
        sx = mx + dx * np.cos(ang) - dy * np.sin(ang)
        sy = my + dx * np.sin(ang) + dy * np.cos(ang)
        sx = sx + np.sin(tau * 1.7 + PHASE[:, None]) * 18 * np.minimum(tau, 3)
        sy = sy + 0.5 * g * tau ** 2 * 12
        floor = ground_y(sx) - 2
        sy = np.minimum(sy, floor)
    return sx, sy, shade * vis * (0.55 + 0.45 * PE), u


def wilt(life):
    droop = smooth(life, 0.75)
    colour = mix(PINK, GREY, droop)
    gravity = 9.8 * droop
    return droop, colour, gravity


def ground_y(x):
    return ROOT_Y + 3 + 40 * (1 - np.exp(-((x - CX) / 260.0) ** 2))


# ----------------------------------------------------------- build particles
def build(life, t):
    droop, colour, gravity = wilt(life)
    fade = (1 - smooth(life, 0.90, 0.09)) * smooth(life, 0.0, 0.015)
    parts = []

    def add(x, y, inten, col):
        x = np.ravel(x)
        inten = np.broadcast_to(inten, np.shape(x)) if np.ndim(inten) == 0 else np.ravel(inten)
        col = np.broadcast_to(col, (x.size, 3)) if np.ndim(col) == 1 else np.reshape(col, (-1, 3))
        parts.append((x, np.ravel(y), inten * fade, col))

    # --- stem
    h = stem_len(life)
    sx, sy = stem(life, t, droop)
    dens = float(np.clip(h / 150, 0.04, 1.0))
    add(CX + sx[None, :] + SJ[:, :, 0], ROOT_Y - sy[None, :] + SJ[:, :, 1],
        0.55 * dens, mix(TEAL, GREY, droop * 0.85))
    top = (CX + sx[-1], ROOT_Y - sy[-1])

    # --- leaves (grow as the stem passes them, droop when wilting)
    for k, (frac, side, length, ang) in enumerate(LEAVES):
        ha = frac * STEM_H
        g = float(smooth(h - ha, 0, 160))
        if g <= 0:
            continue
        fy = ha / STEM_H
        x0 = CX + 14 * math.sin(ha / 80 + t) + droop * 150 * fy ** 2
        y0 = ROOT_Y - ha * (1 - 0.12 * droop * fy)
        L, u, v = length * g, LU[k], LV[k]
        curl = 0.45 + 1.6 * droop
        ax = side * L * u * math.cos(ang)
        ay = L * (u * math.sin(ang) - 0.5 * curl * u * u)
        tx, ty = side * L * math.cos(ang), L * (math.sin(ang) - curl * u)
        n = np.hypot(tx, ty) + 1e-6
        wd = 0.16 * L * np.sin(np.pi * u ** 0.7) * v
        add(x0 + ax + (-ty / n) * wd, y0 - ay - (tx / n) * wd, 0.62,
            mix(TEAL * 0.9, GREY, droop * 0.85))

    # --- the glowing orb: golden seed -> teal tip -> pink bud -> hot core
    orb_col = mix(mix(GOLD, TEAL * 0.5 + 140, smooth(life, 0, 0.1)), CORE,
                  float(smooth(life, 0.2, 0.2)))
    r = (3 + 15 * smooth(life, 0.08, 0.34)) * (1 - 0.75 * smooth(life, 0.42, 0.25))
    add(top[0] + BG[:, 0] * r, top[1] + BG[:, 1] * r, BI * (1 - droop), orb_col)

    # --- petals
    px, py, pi_, pu = bloom(life, t, top, droop)
    tip = mix(colour, VIOLET, RING[:, None] / 6.0 * (1 - droop))
    core = mix(CORE, GREY, droop)
    add(px, py, pi_ * 0.95, mix(core, tip, pu ** 0.7))

    bl = float(smooth(life, 0.42, 0.25)) * (1 - droop * 0.8)
    hc = (top[0], top[1] - 40 * (1 - droop))

    # --- orbit rings that shimmer around the bloom
    rv = float(smooth(life, 0.50, 0.08) * (1 - smooth(life, 0.70, 0.08)))
    if rv > 0:
        for R, sp, tilt in ((150, 0.6, 0.25), (190, -0.45, -0.3)):
            a = RT + t * sp
            ex, ey = R * np.cos(a), R * 0.28 * np.sin(a)
            add(hc[0] + ex * math.cos(tilt) - ey * math.sin(tilt) + rng.normal(0, .5, RN),
                hc[1] + ex * math.sin(tilt) + ey * math.cos(tilt) + rng.normal(0, .5, RN),
                rv * 0.5 * (0.3 + 0.7 * (0.5 + 0.5 * np.sin(a * 3 + t))), PINK * 0.5 + 100)

    # --- pollen sparks drifting off the open flower
    sv = bl * (1 - droop)
    if sv > 0.02:
        age = (t * 0.35 + SP_PH) % 1.0
        add(hc[0] + np.cos(SP_AN) * SP_SP * age, hc[1] - 30 * age + np.sin(SP_AN) * SP_SP * age * 0.7 - 60 * age,
            sv * 0.9 * np.sin(np.pi * age), GOLD)

    # --- floating dust
    dy = (DY - t * DS) % H
    dx = DX + np.sin(t * 0.4 + DP) * 12
    add(dx, dy, 0.22 + 0.18 * np.sin(t * 1.3 + DP * 3), DUST)

    info = dict(top=top, hc=hc, h=h, droop=droop, bl=bl, fade=fade,
                stemfrac=float(h / STEM_H))
    return parts, info


# ---------------------------------------------------------- rendering helpers
def splat(parts):
    x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2][:, None] * p[3] for p in parts])
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    ok = (ix >= 0) & (ix < W - 1) & (iy >= 0) & (iy < H - 1)
    ix, iy, fx, fy, w = ix[ok], iy[ok], fx[ok], fy[ok], w[ok]
    b = ix * H + iy
    idx = np.concatenate([b, b + H, b + 1, b + H + 1])
    ww = np.concatenate([w * ((1 - fx) * (1 - fy))[:, None], w * (fx * (1 - fy))[:, None],
                         w * ((1 - fx) * fy)[:, None], w * (fx * fy)[:, None]])
    acc = np.stack([np.bincount(idx, ww[:, c], minlength=W * H) for c in range(3)], -1)
    return acc.reshape(W, H, 3).astype(np.float32)

def blur(a, r, passes=3):
    for _ in range(passes):
        for ax in (0, 1):
            pad = [(0, 0)] * 3; pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(a, pad), axis=ax, dtype=np.float32)
            cnt = a.shape[ax]
            hi = [slice(None)] * 3; hi[ax] = slice(2 * r + 1, 2 * r + 1 + cnt)
            lo = [slice(None)] * 3; lo[ax] = slice(0, cnt)
            a = (c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)
    return a

def to_surf(a):
    return pygame.surfarray.make_surface(np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8)))

def radial_sprite(w, h, sigma=0.33):
    xx = (np.linspace(-1, 1, w)[:, None]); yy = (np.linspace(-1, 1, h)[None, :])
    d2 = (xx ** 2 + yy ** 2)
    v = np.clip(np.exp(-d2 / (2 * sigma ** 2)) - np.exp(-1 / (2 * sigma ** 2)), 0, 1)
    v = (v / v.max() * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(np.repeat(v[:, :, None], 3, axis=2))

def make_background():
    xx = np.arange(W)[:, None].astype(float); yy = np.arange(H)[None, :].astype(float)
    bg = np.zeros((W, H, 3))
    bg += (np.array([3, 4, 10.]) + np.array([4, 6, 12.]) * (yy / H)[..., None])
    gy = ground_y(xx)
    d = yy - gy
    ground = d >= 0
    gcol = np.array([9, 13, 22.]) * (1 - 0.6 * np.clip(d / (H - ROOT_Y), 0, 1))[..., None]
    rim = np.exp(-np.abs(d) / 3.0)[..., None] * np.array([14, 26, 36.])
    bg = np.where(ground[..., None], gcol, bg) + rim * (np.abs(d) < 14)[..., None]
    star = rng.random((W, H)) > 0.9993                                # a few faint stars
    bg += (star * np.where(yy < ROOT_Y - 60, 1, 0))[..., None] * np.array([40, 45, 70.])
    vig = 1 - 0.55 * (((xx - CX) / (W * 0.75)) ** 2 + ((yy - H * 0.6) / (H * 0.8)) ** 2)
    bg *= np.clip(vig, 0.25, 1)[..., None]
    return pygame.surfarray.make_surface(np.clip(bg, 0, 255).astype(np.uint8))


ASSETS = {}
def init_assets():
    ASSETS["bg"] = make_background()
    big = radial_sprite(256, 256)
    ASSETS["halo"] = pygame.transform.smoothscale(big, (760, 760))
    ASSETS["haze"] = pygame.transform.smoothscale(big, (360, 760))
    ASSETS["pool"] = pygame.transform.smoothscale(big, (620, 150))

def light(screen, sprite, colour, k, center):
    col = tuple(int(c) for c in np.clip(np.asarray(colour) * k, 0, 255))
    s = sprite.copy()
    s.fill(col, special_flags=pygame.BLEND_RGB_MULT)
    screen.blit(s, (int(center[0] - s.get_width() // 2), int(center[1] - s.get_height() // 2)),
                special_flags=pygame.BLEND_RGB_ADD)


# ------------------------------------------------------------------- draw()
def draw(screen, life, t):
    parts, f = build(life, t)
    acc = splat(parts)
    droop, fade, bl, sf = f["droop"], f["fade"], f["bl"], f["stemfrac"]

    # 1. glowing background: dark hill + light pools that follow the flower
    screen.blit(ASSETS["bg"], (0, 0))
    seed = float(1 - smooth(life, 0.0, 0.12))
    light(screen, ASSETS["pool"], mix(TEAL, GOLD, seed), (0.10 + 0.40 * sf + 0.2 * seed) * fade * (1 - droop * 0.6),
          (CX, ROOT_Y + 8))
    light(screen, ASSETS["haze"], TEAL, 0.22 * sf * (1 - droop) * fade,
          (CX, ROOT_Y - f["h"] * 0.5))
    bud = float(smooth(life, 0.05, 0.37)) * (1 - 0.4 * smooth(life, 0.42, 0.2))
    light(screen, ASSETS["halo"], PINK, (0.12 * bud + 0.40 * bl) * fade, f["hc"])

    # 2. sharp particles + 3 bloom layers (tight, medium, wide)
    screen.blit(to_surf(acc), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    a3 = acc.reshape(W // 3, 3, H // 3, 3, 3).sum((1, 3)) / 9
    a12 = a3.reshape(W // 12, 4, H // 12, 4, 3).sum((1, 3)) / 16
    for lay, g in ((blur(a3, 1) * 1.5, 1), (blur(a12, 2) * 1.6, 1), (blur(a12, 6) * 1.8, 1)):
        s = pygame.transform.smoothscale(to_surf(lay), (W, H))
        screen.blit(s, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


def stage(life):
    for lim, name, col in ((0.05, "SEED", GOLD), (0.20, "SPROUT", TEAL), (0.40, "GROWTH", TEAL),
                           (0.70, "BLOOM", PINK), (0.93, "WILT", GOLD), (2, "REBIRTH", GOLD)):
        if life < lim:
            return name, tuple(int(c) for c in col)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("The Whole Life of a Flower")
    init_assets()
    big = pygame.font.SysFont("arial", 26, bold=True)
    mid = pygame.font.SysFont("arial", 15, bold=True)
    clock = pygame.time.Clock()
    life, t, hud, paused = 0.0, 0.0, True, False
    while True:
        dt = clock.tick(60) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                if e.key == pygame.K_h: hud = not hud
                if e.key == pygame.K_r: life = 0.0
        if not paused:
            t += dt
            life += dt / LIFE_SEC
            if life >= 1.0:
                life = 0.0
        draw(screen, life, t)
        if hud:
            title = big.render("THE WHOLE LIFE OF A FLOWER", True, (245, 245, 250))
            screen.blit(title, (W // 2 - title.get_width() // 2, 26))
            name, col = stage(life)
            screen.blit(mid.render(f"DAY {int(life * 91):03d}", True, (210, 210, 220)), (24, H - 44))
            lab = mid.render(name, True, col)
            screen.blit(lab, (W - 24 - lab.get_width(), H - 44))
        pygame.display.flip()


if __name__ == "__main__":
    main()
