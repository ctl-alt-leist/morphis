# Controls and the Playground

This document is the design for interactive control of morphis pictures: a playground where many kinds of mathematical objects, in many kinds of spaces, are moved, reshaped, and evolved by hand or by key, from a panel in a window of its own beside the picture.

The first phase is built: dials and motion, in `morphis.stage`. The visuals the rest stands on are the Scene that draws what it is handed, the `View` and `Projective` family that name a picture, the general bivector exponential, and the `Recording`.

## The One Idea

There is one backend and several front ends. The **stage** holds everything the playground is: the space, the objects in it, which one is selected, the clock, and the messages the user reads. Every front end is a view over the same stage:

- **the panel**, a NiceGUI window with the outline of objects, their dials, and a row of actions;
- **the picture**, the PyVista window, which draws and takes keys, with a one-line status in its corner;
- **a script**, which presses the same commands and sets the same values by name.

A command pressed in the panel, the same key pressed in the picture, and the same command fired from a script change the stage the same way, and the tests check it by driving every front end with one script of keys and comparing the saved state.

Everything else follows from taking that seriously. If a front end decided anything for itself (what a key means, how a value reads, which dials an object has), the front ends would drift apart. So a front end may do only what it alone can do: draw, bind keys, ask for a line of text, show a picker or a file dialog, and close.

The picture is a front end in its own right, not a lesser one. Everything the panel does can be done from the keyboard in the picture alone, with the status line saying where the user is, so the panel is a convenience and never a requirement.

## The Layers

From the bottom up, each layer knows nothing of the ones above it:

1. **The mathematics.** Elements, metrics, transforms, manifolds: morphis as it is.
2. **`morphis.visuals`.** The Scene draws what it is handed and never changes it.
3. **`morphis.stage`, the backend.** The stage, its actors, dials, and journal. No GUI library, no window, no keys. Every change to the stage is an action on it, and every action goes through the journal.
4. **`morphis.app`, what the front ends share.** The `vocabulary` (every command with its key and label), `controls` (what each command does, when it can be pressed, and how its button reads), and `readout` (how every value and object is worded).
5. **The front ends.** `panel.py` (NiceGUI) and `picture.py` (the key bindings and status of the PyVista window). They are the thinnest layer, and the goal is to keep them so.

A feature lands bottom up: an event, its effect in the reducer, an action on the stage, its command in the vocabulary and controls, then each front end. The rule that a widget never changes data is absolute.

## The Stage

The stage holds the space (one `Metric`, so the space's dials mean one thing), the Scene, the tree of actors, the selection, the clock, and the notices. Its actions are what every key, button, and script line ends up calling:

| Action | What it does |
|--------|--------------|
| `add`, `remove` | put an object or an operation on the stage, or take it off |
| `select` | select an entry of the outline, or none |
| `set` | set a dial, by entry and path (`"torus"`, `"shape.major"`, `1.4`) |
| `rename`, `move`, `group` | edit the tree |
| `play`, `pause` | run or hold the clock |
| `stop` | set every rate of the selection to zero |
| `reset` | return the selection's pose to the identity |
| `toggle_frame` | switch the selection between extrinsic and intrinsic |
| `toggle_on`, `bake` | run an operation continuously, or apply it once |
| `save`, `load` | write or read the stage's state |
| `record` | start or stop recording the picture |

The stage also words the **notices**, the running messages the user reads, once for every front end: what happened, and where it matters, what to do next ("select an object to turn it", "reset takes a second r").

```python
g = metric(space=4)

stage = Stage(g, view=Projective())
stage.add("cube", Frame(*basis_vectors(g)), filled=True)
stage.add("torus", clifford_torus, Torus())
stage.add("plane", e1 ^ e2)
stage.select("torus")

stage.run()                       # the picture and the panel, live
```

### The Journal

Every action becomes an event, appended to the stage's journal with the clock time it happened at, and `Stage.apply(event)` is the only reducer. The state the front ends draw from is the journal's events applied in order, and `replay` is that reducer run over a finished journal.

This holds for the clock too. Between two events every rate is constant, and each tick takes the exact step $\exp(-\tfrac{1}{2} Ω Δt)$, which composes exactly, so the poses at every event time are the same whatever the tick rate. A session played by hand at whatever pace the window managed can therefore be replayed off screen at a steady frame rate and recorded, and the recording is the session exactly, without the stutter. Two things fall short of it: an intrinsic velocity outside PGA, whose carrying by the turning pose is taken at the start of each step and so is exact only to first order, and actors with step laws of their own that are not exact flows.

The journal is also where undo would come from, later.

### The Run Loop

Today a script owns both the loop and the clock: `for t in times: update; capture(t)`. With a panel there is no list of times. `run()` loops at wall-clock pace until the picture closes, and each tick it:

1. applies the commands that arrived from the front ends, as events;
2. advances every actor by $Δt$ (times the clock's speed) when the clock is playing;
3. captures, so the Scene redraws;
4. tells the front ends that the stage changed, so each redraws what it reads.

The panel supplies commands, the step laws hold the mathematics, and the Scene draws.

**Scripted use.** The same stage runs with no front end. `stage.advance(dt)` steps it once, values are set by name (`stage["torus"].dials.major = 1.4`, `stage["cube"].motion[1, 3] = 0.5`, where `[1, 3]` addresses the $\mathbf e_{13}$ rate through the metric's labels), and `stage.scene.record(...)` records it, so a figure script and a live session share all of their code.

## The Layout

The panel and the picture keep the same parts in the same order with the same names, and when one changes, the other changes with it in its own form.

```
┌──────────────────────────────────────────────────────────────┐
│ ▶ t = 4.20  ×1   view: perspective, κ = 0.33   torus   ● rec │
├──────────────────────────────────────────────────────────────┤
│ Stage   Library   Journal                                    │
├─────────────────────────┬────────────────────────────────────┤
│ / tor▏                  │ torus  Surface                     │
│ ▾ space                 │ motion   e₁₂ ◯  e₁₃ ◉  …  e₃₄ ◯    │
│   ⟳ turn     Rotation   │          intrinsic                 │
│   ▾ hopf                │                                    │
│       torus   Surface   │ shape    major ◯  minor ◯          │
│     ▸ fibers  Vectors   │ look     color ■  opacity ◯        │
│     cube      Frame     │                                    │
│     plane     Vector    │ invariant planes  0.40 e₁₃ …       │
├─────────────────────────┴────────────────────────────────────┤
│ e₁₃ rate set to 0.40 rad/s                                   │
│ [Add a] [Play ␣] [Stop s] [Reset r] [Intrinsic i] [Hide h]   │
└──────────────────────────────────────────────────────────────┘
```

- **The status bar** along the top reads left to right as the state of the playground: the clock (playing or held, the time, the speed), the view, the selection, and a dot while recording. The picture shows the same line in its corner.
- **The screens strip** under it. The Stage screen is where the playground is worked; Library (saved states and journals) and Journal (the events as they are written) are titled placeholders that fix where those things go before they are built. Moving between screens never adds to the action row.
- **The outline** on the left: the tree of the stage's entries.
- **The inspector** on the right: the dials of the selected entry, then its readouts.
- **Notices and the action row** along the bottom. Every command in the row carries its key, so the keys can be learned off the buttons and the mouse is never required. Toggles are lit while on.

The panel adds only what a mouse makes natural: clicking an entry selects it, dragging a dial turns it, clicking a button presses its key. Nothing is reachable only by mouse.

### The Outline

The outline is a tree of the stage's entries in the order the stage holds them, each showing its name and what kind of object it is. Its root is the **space**: selecting it puts the space's own dials in the inspector (the view's axes, $κ$, shear, and screen turn; the camera's position and orientation; the clock's speed), so the space is worked exactly as any object is. Beneath it are the actors, the operations, and groups of them.

`/` filters the outline by name or kind, the arrows walk it, and Enter selects, so it scales from three objects to three hundred without a row of fixed buttons. Groups fold, reorder, and take members in and out. The outline is a view of the stage's tree, not a structure of its own: renaming, regrouping, and reordering are actions on the stage, and the stage's tree is what saves and loads. The first panel has the tree, filtering, and selection; dragging to reorder and regroup comes after.

Selection is state of the stage, so a script, a key in the picture, or later a click on an object in the picture changes it the same way. The selected object is drawn highlighted.

## Dials

A dial is a parameter with a domain. The domains are the ones the manifold code already has, so an inspector is a chart on a `Domain`:

| Domain | Dial | Value |
|--------|------|-------|
| `Interval(start, stop)` | slider | a real number |
| `Circle()` | knob that wraps | an angle |
| unbounded real | jog wheel | a real number |
| a set of choices | picker | one of them |
| boolean | toggle | on or off |

A group of dials is a `Dials` model, pydantic, knowing nothing of any GUI: it validates, describes itself as a JSON schema, and saves as TOML. Nested groups are sections of the inspector, shown in declaration order. A value is checked against its domain where it is entered and refused with what is allowed ("a number from 0.05 to 1") before it reaches the stage.

Dials are **absolute** or **rates**. An absolute dial is the value itself: a sphere's radius, the view's $κ$. A rate dial is a time derivative: the picture is a function of its history, advanced each tick by $Δt$. Motions are rates.

## Motion

Every object in a space can move rigidly in the ways the space allows. `Motion(metric)` is the group of dials for that, with one rate per basis bivector, and the metric decides the kind of each:

| Plane | $\mathbf B^2$ | Rate |
|-------|---------------|------|
| rotation plane | $< 0$ | an angular rate |
| Lorentzian plane | $> 0$ | a rapidity rate |
| PGA null plane $\mathbf e_{0m}$ | $= 0$ | a velocity |

Together the rates form a bivector $Ω$. An object's pose is a versor $M$, and the drawn object is its rest shape carried by the pose, $x = M x_0 \tilde M$. Each tick advances the pose by the exact step $S = \exp(-\tfrac{1}{2} Ω Δt)$. With $Ω$ held constant this is exactly the flow $\exp(-\tfrac{1}{2} Ω t) M_0$, with no integration error, and $Ω = 0$ holds the object still. $Ω$ is generally not simple, which is why this rests on the general exponential.

In a space without a projective direction, translation is not a versor, so `Motion` also carries a velocity vector $\mathbf v$ that moves the object's origin, $\mathbf o \leftarrow \mathbf o + \mathbf v Δt$. In PGA the translations are already among the null planes, and the extra vector is absent.

Motions are rates because the six rotation planes of $\mathbb{R}^4$ do not commute. Absolute dials $θ^{ab}$ give the pose $\exp(-\tfrac{1}{2} Σ θ^{ab} \mathbf e_{ab})$, which is a fine coordinate on the group but turns in a way no single dial predicts once two are set. A rate dial does what it says: while it is turned up, the object turns in that plane. **Stop** sets $Ω = 0$, and **reset** returns $M$ to the identity.

### Extrinsic and Intrinsic

A step can be applied on either side of the pose, and each object's `Motion` has a toggle between the two:

```math
\begin{aligned}
M &\leftarrow S M & \text{extrinsic} \\
M &\leftarrow M S & \text{intrinsic}
\end{aligned}
```

An **extrinsic** turn acts in the planes of the space: the $\mathbf e_{12}$ dial turns the object in the fixed $\mathbf e_{12}$ plane however the object is already oriented, as on a turntable. An **intrinsic** turn acts in the object's own planes: the $\mathbf e_{12}$ dial turns it in the plane that was $\mathbf e_{12}$ at rest and has since been carried along by the pose, as the pitch, roll, and yaw of an aircraft do. The two agree for a single dial from rest and differ as soon as the object has been turned. Extrinsic is the default. The velocity $\mathbf v$ follows the same toggle: intrinsically it is carried by the pose, $M \mathbf v \tilde M$.

The distinction applies to every operation the dials perform on an object, not only to turns. An extrinsic operation is expressed in the space and acts from the left; an intrinsic one is expressed in the object's own frame and acts from the right.

## Objects

The stage accepts any object at all. What it does with an object depends only on which dials are available for it, and dials are provided per class of object, not per stage.

### Actors

Each object on the stage is held by an **actor**: a name, the object, its dials, and a step law. The actor is what the outline lists and what selection picks; the object is whatever was handed in, untouched.

```python
stage.add("frame", Frame(*basis_vectors(g)))
stage.add("plane", e1 ^ e2)
stage.add("torus", clifford_torus, Torus())
stage.add("ψ", ψ, Evolution(), step=evolve)
```

An object may be something the Scene draws, something built from parameters, or something with no picture at all, such as a field that other actors read. An object no provider recognizes still appears in the outline and can be selected and grouped; it simply has no dials yet.

### Dial Providers

A **dial provider** is a function from an object to a `Dials` group, registered against a class. When an actor is added, its dials are the groups of every provider registered for a class the object is an instance of, from the most general class to the most specific. A new kind of object gets dials by registering a provider for its class, with no change to the stage or any front end.

```python
@provide(Element)
def motion(element: Element) -> Motion:
    return Motion(element.metric)

@provide(Drawable)
def look(element: Drawable) -> Look:
    return Look()                     # color, opacity, visible
```

| Provider | For | Dials |
|----------|-----|-------|
| `motion` | anything with a metric | the rates of its space, the extrinsic/intrinsic toggle |
| `look` | anything the Scene draws | color, opacity, visibility |
| `sampling` | a `Patch` | the resolution of its mesh |
| `motion` | a group | the rates, moving every member together |

### Shapes Built from Parameters

A function from a `Dials` model to an element, the way a manifold is a function of its coordinates. The function's parameters become the actor's `shape` dials, alongside whatever the providers give the element it returns. The stage rebuilds the rest shape only when one of those parameters changes, and the pose still applies on top.

```python
class Torus(Dials):
    major: float = Dial(1.0, Interval(0.2, 3.0))
    minor: float = Dial(0.4, Interval(0.05, 1.0))

def clifford_torus(dials: Torus) -> Surface:
    ...

stage.add("torus", clifford_torus, Torus())
```

### Step Laws

Anything doing something other than rigid motion, such as a state evolving in phase space or a point flowing along a field, gives its own step function. It receives its dials and $Δt$ and writes its object, exactly as the body of today's animation loops does.

```python
class Evolution(Dials):
    energies: tuple[float, float, float] = ...

def evolve(dials: Evolution, ψ: Vector, dt: float) -> None:
    ψ.data[...] = ...

stage.add("ψ", ψ, Evolution(), step=evolve)
```

Rigid motion is one step law among many, the one `Motion` supplies.

### Groups

A group is an actor whose object is its members. The outline shows it as a folder, and its dials come from the providers for groups, so turning a group turns all of its members. Poses compose down the tree, $M = M_{\text{group}} M_{\text{member}}$, so a tangent vector grouped with a sphere turns with the sphere and can still be turned on its own.

## Adding

Objects are added from any front end. In a script, `stage.add` takes an object as it stands. In the panel or the picture, `a` opens the **catalog**: a picker of the kinds of thing that can be added, each under its own letter, filtered by typing like the outline. Choosing one adds it with its default dials, names it (`torus 1`), places it after the selection or inside the selected group, and selects it, so its dials are in the inspector at once.

The catalog is registered the way dial providers are: an entry is a name, a section, and a function from the space to a new object, so a new kind of object is offered by registering it, with no change to any front end.

```python
@catalog("Torus", section="Surfaces")
def torus(space: Metric) -> Actor:
    return Actor(clifford_torus, Torus())
```

| Section | Entries |
|---------|---------|
| Elements | a vector, a bivector, a frame, an expression |
| Surfaces | a plane, a sphere, a torus, a patch from a function |
| Operations | a rotation, a translation, a boost, a reflection |
| Groups | an empty group |

**An expression** asks for a line of text and evaluates it in the space, where the basis vectors are named by their labels: `e1 ^ e3`, `0.5 * (e1 + e2)`, `exp(e12 * pi / 4)`. A physicist's fastest way to put a particular element on the stage is to write it, and the line is kept on the actor, so the expression is its dial and can be edited.

`x` removes the selection, on a second press.

## Operations

An object is acted on by its own dials. An **operation** acts on others: a rotation at a center in a plane, a translation, a boost, a reflection in a hyperplane. An operation is an entry of the outline like any object, with dials of its own (its plane, its center, its angle or rate), drawn in the picture by what it carries (a rotation as its plane's disk at its center), and selected, turned, grouped, and saved as any entry is.

**Scope is position in the tree.** An operation acts on the entries beside it in its group and on everything beneath them. Placed at the root, beside everything, a rotation acts on the whole stage; placed in a group, it acts on that group. Moving an operation in the outline changes what it acts on, and the outline shows what each one reaches.

**On, or applied once.** An operation can be:

- **on**, acting continuously: a rotation with a rate turns its scope while the clock plays, each tick composing $S = \exp(-\tfrac{1}{2} Ω Δt)$ into the poses of everything in scope;
- **applied once** (`b`, for bake, on a second press): its versor is composed into the pose of everything in scope, and the operation is left in place, off, to be applied again or removed.

Operations in one scope apply in outline order, which matters because they do not commute in general. Extrinsic and intrinsic hold here too: an operation is expressed in the space and acts from the left.

An object's own `Motion` is then an operation scoped to that one object. The design keeps them separate entries for clarity in the outline, but they share one machinery: a versor, a scope, a side, and a step.

### Global and Local

The operations above are **global**: one versor $V$ acts on every point alike, which is why it can be composed into poses. A **local** operation is a versor field $V(x)$, different at each point, such as a gauge transformation acting on a field's values point by point. A local operation cannot be folded into a pose, since no single versor describes it. It acts on an object's data: on the points of a mesh, and on the values of a field at each point.

So the general operation is a map on objects, and the global versor is the special case the stage can compose cheaply. Nothing local is designed yet; the point of the split is that the operation interface (scope, order, on or applied once, drawn by what it carries) does not assume a single versor, so local operations, gauge transformations among them, can arrive later without changing the stage or its front ends.

## Keys and the Vocabulary

Every key lives in one place, the `Command` enum in `morphis.app.vocabulary`, with its key, its label, and where it is live. A front end binds and labels commands only from there and never spells a key itself, so changing a key there changes it in the picture, on the panel's buttons, and in the status hints. A test checks that no two keys collide where both are live.

The mapping has a grammar, so a key can be guessed rather than memorized:

- **Digits are directions, named by their labels.** Typing two labels names a plane and moves the inspector's highlight to that plane's dial: `13` is $\mathbf e_{13}$, and `31` is $\mathbf e_{31} = -\mathbf e_{13}$, so it turns the other way. In PGA `0` is the projective direction, as everywhere in morphis. The labels are the metric's, so the keys mean the same thing in every space.
- **Neighbours on the keyboard are neighbours in meaning.** In a Kähler space a shifted digit is the dotted partner: `!` (Shift-1) is $\dot 1$.
- **The arrows walk and turn.** ↑ ↓ walk whichever list has focus, the outline or the inspector; Tab moves focus between them. ← → turn the highlighted dial down and up. Enter on an entry selects it; Enter on a dial asks for a value, checked against its domain where it is typed.
- **Lowercase letters act on the selection.** Space plays and pauses, `s` stops, `r` resets, `i` toggles intrinsic, `h` hides and shows.
- **What cannot be undone takes a second press of the same key.** Reset, remove, and bake. The button turns amber while it waits, and any other command calls the wait off.
- **Commands about the playground itself use the primary modifier**, written `mod-` in a key: ⌘ in a Mac window, Ctrl elsewhere. Save `mod-s`, open `mod-o`, record `mod-e`. They are never bare letters, never buttons, and never letters the system reserves, so nothing of the kind goes off by accident.
- **Screens are reached by `g`, for go,** which opens a picker with each screen under its own letter. Esc goes back to the Stage screen, or lets go of the selection there.
- **Pickers work the same everywhere.** The arrows move the highlight and Enter takes it, or a line's own key takes it at once.
- **A glyph is drawn per front end.** `Command.glyph(modifier)` shows `⌘S` in a Mac window and arrows as `← →`.

| Key | Command | Live |
|-----|---------|------|
| two digits | name a plane, highlighting its dial | an entry with motion selected |
| ← → | turn the highlighted dial | the inspector |
| ↑ ↓ | walk the outline or the inspector | everywhere |
| Tab | move focus between outline and inspector | everywhere |
| Enter | select the entry, or type a dial's value | everywhere |
| `/` | find in the outline | everywhere |
| Space | play or pause | everywhere |
| `s` | stop the selection's rates | an entry with motion selected |
| `r` `r` | reset the selection's pose | an entry with motion selected |
| `i` | extrinsic or intrinsic | an entry with motion selected |
| `h` | hide or show | a drawn entry selected |
| `a` | add, from the catalog | everywhere |
| `x` `x` | remove the selection | an entry selected |
| `o` | turn an operation on or off | an operation selected |
| `b` `b` | apply an operation once | an operation selected |
| `g` | go to a screen | everywhere |
| Esc | let go of the selection, or back to the Stage screen | everywhere |
| `mod-s`, `mod-o` | save, open | everywhere |
| `mod-e` | record | everywhere |

The picture window's own key bindings (PyVista and VTK give `q`, `r`, `s`, `w`, `v`, and others meanings of their own) are cleared, so in the picture every key means what the vocabulary says.

## What a Command Does: Controls

`morphis.app.controls.Controls` decides what pressing each command does, when it can be pressed, and how its button reads (on, off, or waiting on its second press). Front ends bind keys and lay out buttons by calling `press`, `enabled`, and `status`. Controls also holds where the user is: which screen is shown, which list has focus, the highlighted dial, and a plane half named. A command's enabled state is computed from what the stage holds (whether the selection has motion, whether anything is selected), never tracked by a front end. A command that cannot run now is disabled, and pressing it anyway gives a notice saying what to do first.

## Drawing: Panes That Read Plain Data

The panel is built from panes. Each pane has a `read` that pulls what it shows off the stage as plain data (tuples of strings and numbers) and a `draw` that draws that data, and it is redrawn only when what it reads has changed. A stage turning steadily therefore redraws the readouts that change and nothing else. NiceGUI suits this: its views read an object and refresh when it changes.

What a pane shows is worded once, in `morphis.app.readout`: how an object's kind reads in the outline (`Surface`, `Vector, grade 2`), how a rate reads (`e₁₃ 0.40 rad/s`), the invariant planes and angles of a pose from the bivector logarithm. The picture's status line reads the same words. No front end keeps counts or copies of its own; each subscribes to the stage and redraws from what it reads.

## How the Playground Treats Its User

- **Nothing in the loop of playing makes the user stop and type.** Selecting, naming a plane, and turning it are single keys; typed values are there when wanted.
- **Refuse early and say what is allowed.** A value is checked against its domain where it is entered. A command that cannot run is disabled, and pressing it says why.
- **Messages are worded once, in the stage,** as plain sentences that say what happened and, where it matters, what to do next. A front end never words its own message.
- **Never act without being asked.** Nothing is selected, moved, or saved on the user's behalf.
- **Record everything.** The journal keeps every action as it happened, so any session can be replayed, recorded, or picked apart later.

## Testing the Front Ends

One script of keys (`morphis/app/tests/script.py`) drives the picture's key bindings and the panel through NiceGUI's simulated browser, and both must leave the stage in the same saved state. A new command goes into the script in the same piece of work. Because the stage needs no window, most of what a key does is tested headless, by pressing commands through `Controls` against a stage with an off-screen Scene.

## Saving

The state of a stage is its space dials, the tree of actors, and for each actor its dials and its pose. `stage.save("state.toml")` writes all of it in readable TOML, and `Stage.load` restores it, so a view tuned by hand is reproduced exactly by a figure script. The journal is saved beside it when asked for, for replay.

## Module Structure

```
src/morphis/stage/
  dial.py          Dial and its domains, the Dials base model: schema, values by path, TOML
  motion.py        Motion(metric): rates, extrinsic or intrinsic, the exact step
  providers.py     dial providers registered by class, and the common ones
  catalog.py       what can be added, registered by name and section
  operation.py     operations: scope, order, on or applied once
  events.py        the journal's events
  stage.py         Stage and Actor: the space, the tree, selection, the reducer, the run loop
src/morphis/app/
  vocabulary.py    every command, its key, label, and where it is live
  controls.py      what each command does, when, and how its button reads
  readout.py       how every value and object is worded
  panel.py         the NiceGUI panel
  picture.py       the picture window's keys and status line
  tests/script.py  the one script of keys both front ends are driven by
```

`stage` imports `visuals`; `app` imports `stage`; nothing below imports anything above. NiceGUI is an optional dependency (`morphis[gui]`), imported late, so the core library and the scripted stage do not carry a web stack.

## Phases

1. **Dials and motion, headless (built).** `Dial`, `Dials`, `Motion`, `Pose`, values by path. Tested without any window: the extrinsic and intrinsic steps, the exact flow for constant $Ω$, the kinds of rate from the metric.
2. **The stage, scripted.** Actors holding any object, dial providers, the catalog, global operations scoped by the tree, the tree with groups, selection, the journal and its reducer, `advance`, saving, replay into a recording.
3. **The vocabulary, controls, and the picture.** Commands and their keys, `Controls`, the run loop at wall-clock pace, the picture's key bindings and status line. The playground is usable from the keyboard alone at the end of this phase.
4. **The NiceGUI panel.** The panes, the outline with filtering and selection, the inspector rebuilt on selection, the action row, readouts. The key script drives both front ends.
5. **Further.** Dragging in the outline to reorder and regroup; clicking an object in the picture to select it; selecting several objects and showing the dials they share; undo from the journal; the Library and Journal screens; a global-phase dial for Kähler spaces; a slicing-hyperplane control for manifolds; physical knobs behind the same commands; local operations, gauge transformations among them.
