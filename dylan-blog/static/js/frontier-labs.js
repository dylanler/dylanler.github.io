(() => {
  const init = (lab) => {
    if (lab.dataset.ready) return;
    lab.dataset.ready = "true";
    const mode = lab.dataset.frontierMode;
    const control = (name) => lab.querySelector(`[data-control="${name}"]`);
    const output = (name) => lab.querySelector(`[data-output="${name}"]`);
    const readout = lab.querySelector(".frontier-readout");

    if (mode === "camera") {
      const update = () => {
        const pan = +control("pan").value, tilt = +control("tilt").value;
        output("pan").textContent = `${pan}°`; output("tilt").textContent = `${tilt}°`;
        lab.querySelector(".camera-frame").style.transform = `translate(${pan * .7}px, ${tilt * .8}px) rotate(${pan * .025}deg)`;
        const h = pan < -8 ? "right to left pan" : pan > 8 ? "left to right pan" : "locked horizontal axis";
        const v = tilt < -6 ? "rising gaze" : tilt > 6 ? "descending gaze" : "level gaze";
        readout.textContent = `Prompt coordinate: ${h}, ${v}, subject continuity preserved.`;
      };
      lab.querySelectorAll("input").forEach((el) => el.addEventListener("input", update)); update();
    }

    if (mode === "world") {
      const canvas = lab.querySelector("canvas"), ctx = canvas.getContext("2d"); let bodies = [];
      const release = () => bodies.push({x:80 + Math.random()*90,y:35,vx:2 + Math.random()*2.5,vy:0,r:8 + Math.random()*7,h:150 + Math.random()*70});
      for(let i=0;i<7;i++) release();
      const draw = () => {
        const g = +control("gravity").value/100, drag = +control("drag").value/100;
        output("gravity").textContent = g.toFixed(2); output("drag").textContent = drag.toFixed(2);
        ctx.fillStyle="#081318";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle="rgba(101,240,196,.18)";ctx.beginPath();ctx.moveTo(0,270);ctx.lineTo(720,270);ctx.stroke();
        bodies.forEach(b=>{b.vy+=g;b.x+=b.vx;b.y+=b.vy;if(b.y+b.r>270){b.y=270-b.r;b.vy*=-.74;b.vx*=1-drag*.08}if(b.x>740)b.x=-10;ctx.beginPath();ctx.fillStyle=`hsl(${b.h} 75% 65%)`;ctx.shadowBlur=15;ctx.shadowColor=ctx.fillStyle;ctx.arc(b.x,b.y,b.r,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0});requestAnimationFrame(draw);
      };
      lab.querySelector("[data-action='release']").addEventListener("click",release);draw();
    }

    if (mode === "taste") {
      const profiles={maker:[16,12,14,9,"The maker sees constraints as material."],critic:[10,17,12,15,"The critic looks for coherence across choices."],stranger:[13,8,18,18,"The stranger notices what familiarity made invisible."]};
      lab.querySelectorAll("[data-lens]").forEach(btn=>btn.addEventListener("click",()=>{lab.querySelectorAll("button").forEach(b=>b.classList.remove("active"));btn.classList.add("active");const p=profiles[btn.dataset.lens];lab.querySelectorAll(".taste-map circle").forEach((c,i)=>c.setAttribute("r",p[i]));readout.textContent=p[4];}));
    }

    if (mode === "belief") {
      const copy=["One mind, one visible fact.","A belief about another mind enters the scene.","The model must preserve what one person thinks another person believes.","Four layers now coexist. The answer depends on keeping every viewpoint separate."];
      lab.querySelectorAll("[data-depth]").forEach(btn=>btn.addEventListener("click",()=>{const d=+btn.dataset.depth;lab.querySelectorAll("button").forEach(b=>b.classList.remove("active"));btn.classList.add("active");lab.querySelectorAll(".belief-orbit").forEach((o,i)=>{o.classList.toggle("active",i<d);o.classList.toggle("dim",i>=d)});readout.textContent=copy[d-1];}));
      lab.querySelector("[data-depth='1']").click();
    }

    if (mode === "calibration") {
      const update=()=>{const c=+control("confidence").value,e=+control("evidence").value,gap=c-e;output("confidence").textContent=`${c}%`;output("evidence").textContent=`${e}%`;lab.querySelector(".calibration-core strong").textContent=c;lab.querySelector(".calibration-needle").style.transform=`rotate(${-180+c*1.8}deg)`;readout.textContent=Math.abs(gap)<10?"Confidence and evidence are aligned.":gap>0?`Confidence outruns evidence by ${gap} points. This is the danger zone.`:`Evidence leads confidence by ${Math.abs(gap)} points. There may be room to commit.`};lab.querySelectorAll("input").forEach(el=>el.addEventListener("input",update));update();
    }

    if (mode === "memory") {
      const update=()=>{const n=+control("noise").value,d=+control("distance").value;output("noise").textContent=`${n}%`;output("distance").textContent=`${d}×`;lab.querySelector(".memory-noise").style.opacity=.08+n/115;const latent=Math.max(18,96-n*.34-d*1.8),text=Math.max(8,92-n*.55-d*3.2);lab.querySelector("[data-bar='latent']").style.width=`${latent}%`;lab.querySelector("[data-bar='text']").style.width=`${text}%`;};lab.querySelectorAll("input").forEach(el=>el.addEventListener("input",update));update();
    }
  };
  const boot=()=>document.querySelectorAll("[data-frontier-mode]").forEach(init);
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded",boot) : boot();
})();
