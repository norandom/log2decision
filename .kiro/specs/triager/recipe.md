# The triager recipe: find what the search missed

To try this workflow, [open the interactive design fixture](prototype/index.html). It uses twelve traceable originals. The larger figures below illustrate the same investigation on an archive.

An analyst has a large local archive and a broad sign-in search. The question is “Which high-scoring records fell outside that search, and which of those deserve review?” The `triager` skill should answer by reducing named sets, showing the size and cost of each reduction before requesting example text. This is a worked example of the proposed interface. All numbers below are illustrative. [MCP requests](interface.md) · [Scale gate and existing measurements](performance.md)

| Step | Action by the skill or analyst | What comes back |
|---|---|---|
| 1. Establish U | Open the local ZIP with selected entry globs and parser options. Pin its identity. | U has 1,000,000 occurrences and 200,000 distinct originals. Keep both denominators. |
| 2. Establish A and B | A is the top 10,000 by a frozen score, with an explicit tie rule. B is the 5,000 originals matching the content glob `*login*`. Group B into templates if its breadth needs inspection. Preview a change before saving a revised B. | A and B overlap on 2,000 originals. B alone has 3,000. Their union covers 13,000. |
| 3. Choose the Venn region | Create an unbudgeted view of `A − B`. Inspect counts first. Drill by host, time or template for context, or return to the parent. | 8,000 distinct originals remain, 4% of U. The other 192,000 are outside this view. Both producers covered U in this example. |
| 4. Add C where it pays | Run a named recommendation method on those 8,000, recording coverage. Use the product Venn policy with reviewed neighbours, a local GenAI adapter, or issued evidence batches on a public development fixture. An NPU neighbour expansion back into U is an explicit change of scope. | Within the 8,000, C recommends 1,200, passes 6,000 and leaves 800 unevaluated or indeterminate. Outside that scope C was never run. |
| 5. Compare again | Open `(A − B) ∩ C` to inspect the 1,200 recommendations. Inspect the separate 800 unknowns to see what this reduction could hide. Request only bounded examples and their evidence. | The recommended region retains 15% of its parent and 0.6% of U. The unknowns stay visible and eligible for a reserved review. |
| 6. Materialize the analyst work set | In this example allocate 24 distinct originals once: 18 from the recommended region by score, four from the unfamiliar/unevaluated 800, and two seeded audits from the 6,000 assessed passes. Save the ordering, shortfalls, deferrals and selection reasons. | `set_final` has 24 members if all three eligible pools fill. A later visit reopens those exact IDs and evidence. The receipt retains all pool sizes. |

For a broader hunt, compose `A ∪ B ∪ C` before allocation. To ask “what did C alone discover?”, C needs assessed coverage outside both A and B. With step 4's restricted C the answer outside that scope is *unevaluated*. The analyst can deliberately sample that population or commission a wider run and compare the new revision.

The skill's stopping rule is the analyst's question and review budget. It keeps a checkpoint at each reduction and records what was excluded or unassessed. Opening rows is useful when it explains a difference or supplies evidence for review. Repeatedly paging raw rows would squander the archive engine's work. [Membership semantics and terminal drill-down](design.md)
