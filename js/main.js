(function(){
  const yearEl = document.getElementById('year');
  if (yearEl)
    yearEl.textContent = new Date().getFullYear();

  // Tiny animated vector-field placeholder (smooth, non-blocking)
  const c = document.getElementById('vfCanvas');
  if(!c) return;

  const ctx = c.getContext('2d');
  let t = 0;

  function draw(){
    const w = c.width, h = c.height;
    ctx.clearRect(0,0,w,h);

    // Soft grid
    ctx.strokeStyle = '#1b2733';
    ctx.lineWidth = 1;
    for(let x=20; x<w; x+=20){
      ctx.beginPath();
      ctx.moveTo(x,0);
      ctx.lineTo(x,h);
      ctx.stroke();
    }
    for(let y=20; y<h; y+=20){
      ctx.beginPath();
      ctx.moveTo(0,y);
      ctx.lineTo(w,y);
      ctx.stroke();
    }

    // Toy swirling field + time component
    for(let x=20; x<w; x+=20){
      for(let y=20; y<h; y+=20){
        const dx = x - w/2;
        const dy = y - h/2;
        const r2 = dx*dx + dy*dy + 1200;

        const u = (-dy)/Math.sqrt(r2) + 0.6*Math.sin((y+t)/60);
        const v = ( dx)/Math.sqrt(r2) + 0.6*Math.cos((x-t)/60);

        const len = Math.sqrt(u*u + v*v);
        const scale = 8 / Math.max(len, 1e-3);

        ctx.beginPath();
        ctx.moveTo(x,y);
        ctx.lineTo(x + u*scale, y + v*scale);
        ctx.strokeStyle = '#6bd5ff';
        ctx.stroke();
      }
    }

    t += 1.2;
    requestAnimationFrame(draw);
  }

  draw();
})();
