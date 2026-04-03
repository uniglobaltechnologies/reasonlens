import { useEffect, useRef } from "react";
import * as THREE from "three";
import earthTextureUrl from "@/assets/earth-blue-marble.jpg";

const RADIUS = 1.3;

function makeParticleTexture(size = 64): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size * 0.5);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.2, "rgba(111,220,255,0.95)");
  g.addColorStop(1, "rgba(111,220,255,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
  ctx.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

function makeFallbackTexture(): THREE.CanvasTexture {
  const w = 1024, h = 512;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  const grd = ctx.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0, "#0a2742");
  grd.addColorStop(1, "#07162a");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, w, h);
  ctx.globalAlpha = 0.08;
  for (let y = 0; y < h; y += 8) {
    ctx.fillStyle = "#6fdcff";
    ctx.fillRect(0, y, w, 2);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function addParticleField(globeGroup: THREE.Group, count: number): THREE.Group {
  const particles: THREE.Vector3[] = [];
  const positions = new Float32Array(count * 3);
  const v = new THREE.Vector3();
  const particleRadius = RADIUS * 1.14;

  for (let i = 0; i < count; i++) {
    v.randomDirection().multiplyScalar(particleRadius + (Math.random() - 0.5) * 0.08);
    positions[i * 3] = v.x;
    positions[i * 3 + 1] = v.y;
    positions[i * 3 + 2] = v.z;
    particles.push(new THREE.Vector3(v.x, v.y, v.z));
  }

  const starGroup = new THREE.Group();

  const dotsGeo = new THREE.BufferGeometry();
  dotsGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const dots = new THREE.Points(
    dotsGeo,
    new THREE.PointsMaterial({
      size: 0.012,
      map: makeParticleTexture(32),
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
  starGroup.add(dots);

  const threshold = 0.38;
  const lineVerts: number[] = [];
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      if (particles[i].distanceTo(particles[j]) < threshold) {
        lineVerts.push(particles[i].x, particles[i].y, particles[i].z);
        lineVerts.push(particles[j].x, particles[j].y, particles[j].z);
      }
    }
  }

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
  const lines = new THREE.LineSegments(
    lineGeo,
    new THREE.LineBasicMaterial({
      color: 0x4ddbff,
      transparent: true,
      opacity: 0.1,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
  starGroup.add(lines);

  globeGroup.add(starGroup);
  return starGroup;
}

function addAtmosphere(globeGroup: THREE.Group, segments: number) {
  const vertexShader = `
    varying vec3 vNormal;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  // Teal-warm glow shell
  const frag1 = `
    varying vec3 vNormal;
    uniform float warmMix;
    void main() {
      float rim = pow(1.0 - abs(vNormal.z), 3.0);
      vec3 teal = vec3(0.12, 0.85, 1.0);
      vec3 warm = vec3(1.0, 0.72, 0.62);
      vec3 hue = mix(teal, warm, warmMix);
      gl_FragColor = vec4(hue, 1.0) * rim * 0.35;
    }
  `;
  const shell1 = new THREE.Mesh(
    new THREE.SphereGeometry(RADIUS * 1.1, segments, segments),
    new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: frag1,
      uniforms: { warmMix: { value: 0.58 } },
      side: THREE.BackSide,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  globeGroup.add(shell1);

  // Directional warm glow
  const frag2 = `
    varying vec3 vNormal;
    uniform vec3 sunDir;
    void main() {
      float rim = pow(1.0 - abs(vNormal.z), 3.0);
      float dir = max(0.0, dot(normalize(vNormal), normalize(sunDir)));
      float intensity = rim * pow(dir, 1.5);
      gl_FragColor = vec4(1.0, 0.58, 0.28, 1.0) * intensity * 0.45;
    }
  `;
  const shell2 = new THREE.Mesh(
    new THREE.SphereGeometry(RADIUS * 1.13, segments, segments),
    new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader: frag2,
      uniforms: { sunDir: { value: new THREE.Vector3(0.7, 0.25, 1) } },
      side: THREE.BackSide,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  globeGroup.add(shell2);
}

export default function GlobeCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Detect lite mode
    const conn = (navigator as any).connection;
    const prefersLite =
      (conn && (conn.saveData || /2g/.test(conn.effectiveType || ""))) ||
      (window.matchMedia && window.matchMedia("(prefers-reduced-data: reduce)").matches);
    const useLite = prefersLite;
    const pixelRatio = useLite ? 1 : Math.min(window.devicePixelRatio || 1, 2);
    const targetFPS = useLite ? 30 : 60;
    const atmoSegments = useLite ? 48 : 64;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    camera.position.set(0, 0, 6);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x1a1a1a, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.045);
    dir.position.set(-2, 1, 1);
    scene.add(dir);

    const globeGroup = new THREE.Group();
    scene.add(globeGroup);
    const orbitGroup = new THREE.Group();
    globeGroup.add(orbitGroup);

    function fitCamera() {
      const w = canvas!.clientWidth;
      const h = canvas!.clientHeight || 420;
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    function buildWith(tex: THREE.Texture | null) {
      if (tex) {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = useLite ? 4 : 8;
      }

      const earthMat = new THREE.MeshPhongMaterial({
        map: tex || makeFallbackTexture(),
        color: 0xffffff,
        shininess: 12,
      });
      const sphereDetail = useLite ? 64 : 128;
      const earth = new THREE.Mesh(
        new THREE.SphereGeometry(RADIUS, sphereDetail, sphereDetail),
        earthMat
      );
      orbitGroup.add(earth);

      const nightShell = new THREE.Mesh(
        new THREE.SphereGeometry(RADIUS * 1.001, 64, 64),
        new THREE.MeshPhongMaterial({
          color: 0x061233,
          emissive: 0x0b1029,
          emissiveIntensity: 0.25,
          transparent: true,
          opacity: 0.18,
        })
      );
      orbitGroup.add(nightShell);

      addAtmosphere(globeGroup, atmoSegments);
      const starGroup = addParticleField(globeGroup, useLite ? 150 : 250);

      fitCamera();

      let ro: ResizeObserver | null = null;
      if ("ResizeObserver" in window && canvas!.parentElement) {
        ro = new ResizeObserver(fitCamera);
        ro.observe(canvas!.parentElement);
      }
      window.addEventListener("resize", fitCamera);

      let rafId: number;
      let lastTime = 0;
      let acc = 0;
      const interval = 1000 / targetFPS;

      function animate(now: number) {
        rafId = requestAnimationFrame(animate);
        if (!lastTime) lastTime = now;
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        acc += now - lastTime;
        lastTime = now;
        if (acc < interval) return;
        acc = 0;
        orbitGroup.rotation.y += 0.14 * dt;
        starGroup.rotation.y -= 0.06 * dt;
        renderer.render(scene, camera);
      }
      rafId = requestAnimationFrame(animate);

      cleanupRef.current = () => {
        cancelAnimationFrame(rafId);
        window.removeEventListener("resize", fitCamera);
        ro?.disconnect();
        renderer.dispose();
      };
    }

    const loader = new THREE.TextureLoader();
    loader.load(
      earthTextureUrl,
      (tex) => buildWith(tex),
      undefined,
      () => buildWith(null)
    );

    return () => {
      cleanupRef.current?.();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full block"
      style={{ background: "transparent" }}
    />
  );
}
