(function initTruthLensBackground(rootDoc) {
    "use strict";

    var doc = rootDoc || document;
    var win = doc.defaultView || window;

    if (win.__tlBgInit) return;
    win.__tlBgInit = true;

    var reduceMotion = win.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) return;

    var canvas = doc.getElementById("tl-bg-canvas");
    if (!canvas) return;

    var ctx = canvas.getContext("2d", { alpha: true });
    var width = 0;
    var height = 0;
    var particles = [];
    var mouseX = 0.5;
    var mouseY = 0.5;
    var rafId = null;

    var PARTICLE_COUNT = 72;
    var CONNECT_DIST = 120;

    function resize() {
        width = win.innerWidth;
        height = win.innerHeight;
        canvas.width = width;
        canvas.height = height;
    }

    function rand(min, max) {
        return min + Math.random() * (max - min);
    }

    function initParticles() {
        particles = [];
        for (var i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                z: Math.random(),
                vx: rand(-0.25, 0.25),
                vy: rand(-0.35, 0.35),
                hue: Math.random() > 0.5 ? 265 : 190
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);

        var parallaxX = (mouseX - 0.5) * 18;
        var parallaxY = (mouseY - 0.5) * 12;

        for (var i = 0; i < particles.length; i++) {
            var p = particles[i];
            p.x += p.vx + parallaxX * 0.002 * p.z;
            p.y += p.vy + parallaxY * 0.002 * p.z;

            if (p.x < -20) p.x = width + 20;
            if (p.x > width + 20) p.x = -20;
            if (p.y < -20) p.y = height + 20;
            if (p.y > height + 20) p.y = -20;

            var radius = 0.8 + p.z * 2.2;
            var alpha = 0.15 + p.z * 0.45;

            ctx.beginPath();
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = "hsla(" + p.hue + ", 80%, 68%, " + alpha + ")";
            ctx.fill();
        }

        for (var a = 0; a < particles.length; a++) {
            for (var b = a + 1; b < particles.length; b++) {
                var pa = particles[a];
                var pb = particles[b];
                var dx = pa.x - pb.x;
                var dy = pa.y - pb.y;
                var dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONNECT_DIST) {
                    var lineAlpha = (1 - dist / CONNECT_DIST) * 0.12;
                    ctx.beginPath();
                    ctx.moveTo(pa.x, pa.y);
                    ctx.lineTo(pb.x, pb.y);
                    ctx.strokeStyle = "rgba(123, 92, 240, " + lineAlpha + ")";
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }
        }

        rafId = win.requestAnimationFrame(draw);
    }

    function onMouseMove(e) {
        mouseX = e.clientX / Math.max(width, 1);
        mouseY = e.clientY / Math.max(height, 1);
    }

    function start() {
        resize();
        initParticles();
        if (rafId) win.cancelAnimationFrame(rafId);
        draw();
    }

    win.addEventListener("resize", start);
    win.addEventListener("mousemove", onMouseMove, { passive: true });
    start();
})(typeof window !== "undefined" && window.parent && window.parent.document ? window.parent.document : document);
