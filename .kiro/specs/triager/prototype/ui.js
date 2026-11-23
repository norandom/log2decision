"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const make = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (cls) node.className = cls;
    return node;
  };
  const button = (label, action, cls) => {
    const node = make("button", label, cls);
    node.type = "button";
    node.addEventListener("click", () => attempt(action));
    return node;
  };
  const stateText = {in:"in", out:"out", not_evaluated:"unevaluated"};
  const symbols = {in:"+", out:"−", not_evaluated:"?"};
  const regionNames = {a_minus_b:"A − B · scored only", both:"A ∩ B · overlap", b_minus_a:"B − A · search only", neither:"Neither A nor B", recommended:"C · recommended", pass:"C · assessed pass", unknown:"C · unevaluated"};
  const dialog = byId("dialog");
  const sessions = [], frames = new Map(), receipts = new Map();
  let driver, active, root, base, final;

  function attempt(action) {
    try { return action(); }
    catch (error) { status("Operation failed. " + error.message); }
  }
  function status(message) {
    byId("status-text").textContent = message;
    byId("trace-count").textContent = sessions.reduce((count, s) => count + s.trace.length, 0);
  }
  function label(handle, fallback) {
    if (handle === root) return "Archive scope";
    if (handle === base) return "Scored outside the search";
    if (handle === final) return "Final review work set";
    return fallback || handle;
  }
  function frame(handle, parent = null, title) {
    if (!frames.has(handle)) frames.set(handle, {handle, parent, title:label(handle, title), query:"", mode:"literal", selected:null});
    return frames.get(handle);
  }
  function pathTo(handle) {
    const path = [];
    let f = frames.get(handle);
    while (f) { path.unshift(f); f = frames.get(f.parent); }
    return path;
  }
  function remember() {
    if (!active) return;
    const f = frames.get(active);
    f.query = byId("search-text").value;
    f.mode = byId("search-mode").value;
    f.tableLeft = byId("records-wrap").scrollLeft;
    f.scrollTop = document.scrollingElement.scrollTop;
    const target = document.activeElement;
    if (target?.id) f.focusSelector = "#" + CSS.escape(target.id);
    else if (target instanceof HTMLElement) {
      const entry=Object.entries(target.dataset)[0];
      if (entry) f.focusSelector = "[data-"+entry[0]+'="'+CSS.escape(entry[1])+'"]';
    }
  }
  function navigate(handle, parent, title) {
    remember();
    const f = frame(handle, parent, title);
    if (!f.data) f.data = driver.inspect(handle);
    active = handle;
    byId("search-text").value = f.query;
    byId("search-mode").value = f.mode;
    render(f);
    byId("records-wrap").scrollLeft = f.tableLeft || 0;
    if (f.focusSelector) document.querySelector(f.focusSelector)?.focus({preventScroll:true});
    else byId("view-title").focus({preventScroll:true});
    document.scrollingElement.scrollTop = f.scrollTop || 0;
    status("View " + handle + " · exact fixture counts");
  }
  function drill(kind, value, title) {
    const parent = active;
    const next = driver.drill(parent, kind, value);
    navigate(next, parent, title);
  }
  function modal(title, build) {
    byId("dialog-title").textContent = title;
    const body = byId("dialog-body");
    body.replaceChildren();
    build(body);
    if (!dialog.open) dialog.showModal();
  }
  function entries(parts) {
    const dl = make("dl");
    for (const [key,value] of parts) dl.append(make("dt",key),make("dd",value));
    return dl;
  }
  function closeThen(action) {
    dialog.close();
    action();
  }
  function render(f) {
    const data = f.data, c = data.counts, oc = data.occurrence_counts;
    byId("view-title").textContent = f.title;
    byId("back").disabled = !f.parent;
    byId("allocate").textContent = final ? "Open review work set" : "Allocate review slots";
    byId("allocate").disabled = !!data.allocated;
    const bc = byId("breadcrumbs"); bc.replaceChildren();
    const lineage = pathTo(active);
    for (const [i,ancestor] of lineage.entries()) {
      if (i) bc.append(make("span","/","slash"));
      if (ancestor.handle === active) bc.append(make("span",ancestor.title,"current"));
      else bc.append(button(ancestor.title,() => navigate(ancestor.handle),"undressed"));
    }
    const metrics = byId("metrics"); metrics.replaceChildren();
    const items = [
      [c.in + " / " + c.parent, f.parent ? "parent originals" : "snapshot originals"],
      [oc.in + " / " + oc.parent, "occurrences"],
      [c.retained_share === null ? "—" : Math.round(c.retained_share * 1000) / 10 + "%", "retained"],
      [c.out, data.allocated ? "outside work set" : "excluded"],
      [c.unknown,"indeterminate"]
    ];
    items.forEach(([value,unit],i) => {
      const p = make("span",undefined,"metric" + (i===0?" primary":""));
      p.append(make("strong",value)," "+unit);
      metrics.append(p);
    });
    const regions = byId("regions"); regions.replaceChildren();
    for (const family of ["ab","c"]) {
      regions.append(make("div",family==="ab"?"Score × search":"Recommendation coverage","region-heading"));
      for (const r of data.regions) {
        const group=r.family || (["recommended","pass","unknown"].includes(r.key)?"c":"ab");
        if (group!==family) continue;
        const name = regionNames[r.key] || r.label;
        const b = facet(name,r.count,"distinct originals",() => drill("region",r.key,name));
        b.dataset.region = r.key;
        regions.append(b);
      }
    }
    const hosts = byId("hosts"); hosts.replaceChildren();
    for (const h of data.hosts) {
      const b = facet(h.value,h.count,"occurrences",() => drill("host",h.value,"Host · "+h.value));
      b.dataset.host = h.value;
      hosts.append(b);
    }
    if (!data.hosts.length) hosts.append(make("p","No hosts in this view.","empty"));
    const bins = byId("bins"); bins.replaceChildren();
    const maximum = Math.max(1,...data.bins.map(h=>h.count));
    for (const bin of data.bins) {
      const hour = bin.value.slice(11,16);
      const b = button("",() => drill("hour",bin.value,"Hour · "+hour+" UTC"),"bin");
      b.dataset.hour = bin.value;
      b.setAttribute("aria-label",hour+" UTC, "+bin.count+" occurrences. Drill into this hour.");
      const bar = make("span",undefined,"bar");
      bar.style.height = (5 + Math.round(36*bin.count/maximum))+"px";
      b.append(bar,make("span",hour + " · " + bin.count,"bin-label"));
      bins.append(b);
    }
    if (!data.bins.length) bins.append(make("p","No occurrences in this view.","empty"));
    renderRecords(f);
  }
  function facet(label, count, unit, fn) {
    const b = button("",fn,"facet-row");
    b.append(make("span",label),make("span",count,"count"));
    b.title = count+" "+unit+" in this view";
    b.setAttribute("aria-label",label+", "+count+" "+unit);
    b.disabled = count===0;
    return b;
  }
  function renderRecords(f) {
    const records = f.data.records, body = byId("records"); body.replaceChildren();
    byId("record-count").textContent = records.length + " shown / " + f.data.counts.in;
    for (const r of records) {
      const tr = make("tr");
      if (f.selected===r.id) tr.className="selected";
      const first = make("td"), b = button("",() => openEvidence(r),"record-button");
      b.dataset.record=r.id;
      b.append(make("span",r.id,"record-code"),make("span",r.text));
      if (r.allocation) b.append(make("span",r.allocation+" slot","allocation"));
      first.append(b);
      tr.append(first,make("td",r.score.toFixed(2)));
      for (const s of ["a","b","c"]) {
        const val=r.membership[s], td=make("td",symbols[val],val==="not_evaluated"?"member-unknown":"member-"+val);
        td.title=s.toUpperCase()+" "+stateText[val];
        tr.append(td);
      }
      tr.append(make("td",r.occurrences));
      body.append(tr);
    }
    if (!records.length) {
      const tr=make("tr"), td=make("td","No original records match. Back returns to the parent view.","empty");
      td.colSpan=6; tr.append(td); body.append(tr);
    }
  }
  function openEvidence(row) {
    const result=driver.evidence(active,row.id);
    const f=frames.get(active);
    f.selected=row.id;
    // Preserve the invoking button so closing the dialog restores keyboard focus.
    for (const tr of byId("records").rows) tr.classList.toggle("selected",tr.querySelector("button")?.dataset.record===row.id);
    modal("Evidence · "+row.id,body=>{
      body.append(make("p",result.text,"message"));
      body.append(entries([
        ["Source view",f.title+" ("+active+")"],["Text score",result.score.toFixed(2)],
        ["Membership",Object.entries(result.membership).map(([s,val])=>s.toUpperCase()+" "+stateText[val]).join(" · ")],
        ["Recommendation",result.reason],["Allocation",result.allocation || "No slot assigned in this view"]
      ]));
      body.append(make("h3","Occurrences in this view"));
      const list=make("ul",undefined,"list");
      for (const e of result.locations) list.append(make("li",e.time+" · "+e.host+" · "+e.locator));
      body.append(list);
    });
    status("Read evidence for "+row.id+" in "+active);
  }
  function allocationDialog() {
    if (!frames.has(base)) frame(base,root);
    if (final) { navigate(final,base,"Final review work set"); return; }
    modal("Allocate from the A − B branch",body=>{
      body.append(make("p","This fixture allocates four original records from the full scored-outside-search branch. Field and time filters on a child stay on that child."));
      body.append(entries([["2 general","Highest-scoring C recommendations"],["1 unfamiliar","C unevaluated, ordered by record ID"],["1 audit","Seeded sample from C assessed passes"]]));
      const b=button("Allocate 4 from A − B",()=>closeThen(()=>{
        final=driver.allocate(); navigate(final,base,"Final review work set");
        status("Allocated four distinct records once. Reopen this work set to inspect its reasons.");
      }),"accent"); b.id="confirm-allocation"; body.append(b);
    });
  }
  function showBranches() {
    modal("Investigation branches",body=>{
      const list=make("ul",undefined,"list");
      for (const f of frames.values()) {
        const li=make("li"), b=button("",()=>closeThen(()=>navigate(f.handle)));
        b.append(make("span",pathTo(f.handle).map(x=>x.title).join(" / ")),make("small",f.handle===active?"active":f.handle));
        b.dataset.branch=f.handle; li.append(b); list.append(li);
      }
      body.append(list);
    });
  }
  function saveView() {
    const receipt=driver.save(active);
    receipts.set(receipt,{title:frames.get(active).title});
    byId("saved-count").textContent=receipts.size;
    status("Saved "+receipt+" · Open Handoffs to resume in a new fixture client.");
  }
  function showHandoffs() {
    modal("Saved handoffs",body=>{
      body.append(make("p","Open a saved receipt in a fresh fixture client. Receipts are available for this browser session until the page is reloaded.","hint"));
      if (!receipts.size) { body.append(make("p","Save any view or the final work set first.","empty")); return; }
      const list=make("ul",undefined,"list");
      for (const [receipt,saved] of receipts) {
        const b=button("",()=>closeThen(()=>{
          const peer=TriagerDemo.create(driver.store);
          const handle=peer.reopen(receipt);
          const lineage=peer.call("triager_inspect",{v:1,handle,view:"lineage"}).path;
          driver=peer; sessions.push(peer);
          active=null;
          frames.clear();
          for (const ancestor of lineage) frame(ancestor.handle,ancestor.parent,ancestor.title);
          navigate(handle);
          status("Client "+sessions.length+" reopened "+receipt+" · no producer rerun");
        }));
        b.dataset.receipt=receipt;
        b.append(make("span",saved.title),make("small",receipt));
        const li=make("li"); li.append(b); list.append(li);
      }
      body.append(list);
    });
  }
  function showTrace() {
    modal("MCP operation trace",body=>{
      body.append(make("p","Fixture calls are shown newest first. Expand a call to inspect its arguments and result.","hint"));
      for (let n=sessions.length-1;n>=0;n--) {
        const history=sessions[n].trace;
        for (let i=history.length-1;i>=0;i--) {
          const call=history[i], row=make("details",undefined,"trace-entry");
          row.append(make("summary","Client "+(n+1)+" · "+call.tool+(call.args.view?" · "+call.args.view:"")));
          row.addEventListener("toggle",()=>{
            if (row.open && !row.querySelector("pre")) row.append(make("pre",JSON.stringify(call,null,2)));
          });
          body.append(row);
        }
      }
    });
  }

  byId("close-dialog").addEventListener("click",()=>dialog.close());
  byId("search-form").addEventListener("submit",event=>{
    event.preventDefault();
    const query=byId("search-text").value;
    if (!query.trim()) { byId("search-text").focus(); status("Enter a literal or glob to refine this view."); return; }
    attempt(()=>drill(byId("search-mode").value,query,"Search · "+query));
  });
  byId("search-text").maxLength=120;
  byId("search-text").addEventListener("keydown",event=>{
    if (event.key==="Enter" && !event.shiftKey) { event.preventDefault(); byId("search-form").requestSubmit(); }
  });
  byId("search-height").addEventListener("input",event=>{byId("search-text").rows=Number(event.target.value);});
  byId("sidebar-width").addEventListener("input",event=>document.documentElement.style.setProperty("--sidebar-width",event.target.value+"px"));
  byId("save").addEventListener("click",()=>attempt(saveView));
  byId("handoffs").addEventListener("click",showHandoffs);
  byId("trace").addEventListener("click",showTrace);
  byId("branches").addEventListener("click",showBranches);
  byId("allocate").addEventListener("click",()=>attempt(allocationDialog));
  byId("back").addEventListener("click",()=>attempt(()=>navigate(frames.get(active).parent)));
  document.addEventListener("keydown",event=>{
    if (dialog.open || event.ctrlKey || event.metaKey || event.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) return;
    if (event.key==="/") { event.preventDefault(); byId("search-text").focus(); }
    if (event.key==="Escape" && frames.get(active)?.parent) { event.preventDefault(); byId("back").click(); }
  });
  attempt(()=>{
    driver=TriagerDemo.create(); sessions.push(driver);
    const boot=driver.start(); root=boot.snapshot; base=boot.difference;
    frame(root);
    navigate(base,root);
    status("Prepared A − B opened · C assessed within this branch");
  });
})();
