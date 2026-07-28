/* QuOptuna site-wide animated background: drifting star particles + a cursor
   halo, rendered behind all content. Injected on every page via the Starlight
   `head` config. The custom landing page renders its own richer background, so
   this script no-ops there (detected by the .qo-hero element / #qo-space). */
(function () {
  "use strict";
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  function init() {
    // Landing page already has its own background — don't double up.
    if (document.querySelector(".qo-hero") || document.getElementById("qo-space")) return;
    if (document.getElementById("qo-bg-space")) return;

    var space = document.createElement("canvas");
    space.id = "qo-bg-space";
    space.setAttribute("aria-hidden", "true");
    Object.assign(space.style, {
      position: "absolute", top: "0", left: "0", width: "100%",
      zIndex: "0", pointerEvents: "none", opacity: "0.7",
    });

    var halo = document.createElement("div");
    halo.id = "qo-bg-halo";
    halo.setAttribute("aria-hidden", "true");
    Object.assign(halo.style, {
      position: "fixed", top: "0", left: "0", width: "520px", height: "520px",
      margin: "-260px 0 0 -260px", borderRadius: "50%", zIndex: "0",
      pointerEvents: "none", opacity: "0", transition: "opacity .6s cubic-bezier(.2,.7,.3,1)",
      background:
        "radial-gradient(closest-side, color-mix(in srgb, var(--qo-quantum,#a78bfa) 12%, transparent), transparent 62%)," +
        "radial-gradient(closest-side at 62% 62%, color-mix(in srgb, var(--qo-classical,#fb923c) 9%, transparent), transparent 60%)",
      filter: "blur(14px)", willChange: "transform",
    });

    document.body.insertBefore(space, document.body.firstChild);
    document.body.insertBefore(halo, document.body.firstChild);

    var sx = space.getContext("2d");
    var DPR = Math.min(window.devicePixelRatio || 1, 2);
    var W = 0, H = 0, stars = [];
    var cvar = function (v) {
      return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#a78bfa";
    };
    var docHeight = function () {
      return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, window.innerHeight);
    };
    function resize() {
      W = window.innerWidth; H = docHeight();
      space.style.height = H + "px";
      space.width = W * DPR; space.height = H * DPR; sx.setTransform(DPR, 0, 0, DPR, 0, 0);
      var target = Math.round(W * H / 16000);
      stars.length = 0;
      for (var i = 0; i < target; i++) {
        var big = Math.random() < 0.14;
        var r = big ? (2 + Math.random() * 2) : (0.3 + Math.random() * 1.1);
        var s = {
          x: Math.random() * W, y: Math.random() * H, r: r, hx: 0, hy: 0,
          vx: (Math.random() - 0.5) * 0.05, vy: (Math.random() - 0.5) * 0.05,
          tw: Math.random() * 6.28, tws: (0.008 + Math.random() * 0.015) * (big ? 0.7 : 1),
          c: Math.random() < 0.5,
        };
        s.hx = s.x; s.hy = s.y; stars.push(s);
      }
    }
    var wells = [];
    document.addEventListener("click", function (e) {
      if (e.target && e.target.closest && e.target.closest("a, button, input, textarea, select, label, [role=button]")) return;
      wells.push({ x: e.pageX, y: e.pageY, life: 1 });
      if (wells.length > 4) wells.shift();
    });
    function frame() {
      if (!W) { requestAnimationFrame(frame); return; }
      var q = cvar("--qo-quantum"), c = cvar("--qo-classical");
      sx.clearRect(0, 0, W, H);
      for (var w = 0; w < wells.length; w++) wells[w].life -= 0.006;
      for (var k = wells.length - 1; k >= 0; k--) if (wells[k].life <= 0) wells.splice(k, 1);
      for (var i = 0; i < stars.length; i++) {
        var s = stars[i];
        for (var j = 0; j < wells.length; j++) {
          var wl = wells[j];
          var dx = wl.x - s.x, dy = wl.y - s.y, d2 = dx * dx + dy * dy, R = 260;
          if (d2 < R * R) { var d = Math.sqrt(d2) || 1, pull = (1 - d / R) * wl.life * 0.045;
            s.vx += (dx / d) * pull; s.vy += (dy / d) * pull; }
        }
        s.vx += (s.hx - s.x) * 0.0004; s.vy += (s.hy - s.y) * 0.0004;
        s.x += s.vx; s.y += s.vy; s.vx *= 0.97; s.vy *= 0.97; s.tw += s.tws;
        if (s.x < 0) s.x = W; if (s.x > W) s.x = 0;
        if (s.y < 0) s.y = H; if (s.y > H) s.y = 0;
        sx.globalAlpha = 0.22 + (Math.sin(s.tw) * 0.5 + 0.5) * 0.45;
        sx.fillStyle = s.c ? q : c;
        sx.beginPath(); sx.arc(s.x, s.y, s.r, 0, 6.283); sx.fill();
      }
      sx.globalAlpha = 1;
      requestAnimationFrame(frame);
    }
    window.addEventListener("resize", resize);
    resize(); requestAnimationFrame(frame);
    window.addEventListener("load", resize);
    setTimeout(resize, 600);

    if (window.matchMedia("(pointer:fine)").matches) {
      var hx = window.innerWidth / 2, hy = window.innerHeight / 2, tx = hx, ty = hy;
      window.addEventListener("pointermove", function (e) { tx = e.clientX; ty = e.clientY; halo.classList.add("on"); halo.style.opacity = "0.8"; });
      window.addEventListener("pointerleave", function () { halo.style.opacity = "0"; });
      (function loop() {
        hx += (tx - hx) * 0.1; hy += (ty - hy) * 0.1;
        halo.style.transform = "translate3d(" + hx + "px," + hy + "px,0)";
        requestAnimationFrame(loop);
      })();
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
