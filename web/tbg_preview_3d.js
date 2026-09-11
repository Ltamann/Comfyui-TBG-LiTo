import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const viewerControllers = new WeakMap();
const guardedViewers = new WeakSet();
const guardedViewerContainers = new WeakSet();
const backgroundLoads = new WeakMap();

const TBG_NODE_ID = "Trellis2Load3D";
const TBG_DISPLAY_NAME = "Composit 3D mesh o Splat on Background";
const CAPTURE_SCALE = 2;

function guardViewerEvents(container) {
    if (!container || guardedViewerContainers.has(container)) return;
    guardedViewerContainers.add(container);
    container.addEventListener("pointerdown", (event) => event.stopPropagation());
}

function disableViewerInertia(load3d) {
    const controls = load3d.controlsManager?.controls;
    if (!controls) return;
    controls.enableDamping = false;
    controls.update();
}

function snapshotViewerCamera(node, load3d) {
    const camera = node.properties?.["Camera Config"] || {};
    const state = load3d.getCameraState?.() || camera.state;
    const snapshot = {
        ...camera,
        cameraType: load3d.getCurrentCameraType?.() || camera.cameraType,
        fov: load3d.cameraManager?.perspectiveCamera?.fov ?? camera.fov,
        state: state ? structuredClone(state) : state,
    };
    node.properties ||= {};
    node.properties["Camera Config"] = snapshot;
    return snapshot;
}

function restoreViewerCamera(node, load3d, camera) {
    if (!camera) return;
    if (camera.cameraType) load3d.toggleCamera(camera.cameraType);
    if (camera.fov != null) load3d.setFOV(camera.fov);
    if (camera.state) load3d.setCameraState(structuredClone(camera.state));
    load3d.forceRender?.();
    node.properties ||= {};
    node.properties["Camera Config"] = camera;
}

function guardViewerBackground(node, load3d) {
    if (guardedViewers.has(load3d)) return;
    guardedViewers.add(load3d);
    const scene = load3d.getSceneManager();
    const renderBackground = scene.renderBackground;
    scene.renderBackground = function (...args) {
        if (this.backgroundTexture) {
            // Spark writes pixel-store state outside Three's cache, including
            // when another node uses the shared renderer.
            const gl = this.renderer.getContext();
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, this.backgroundTexture.flipY);
            this.renderer.state.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, this.backgroundTexture.flipY);
        }
        return renderBackground.apply(this, args);
    };

    const setBackgroundImage = load3d.setBackgroundImage.bind(load3d);
    let pending = Promise.resolve();
    let pendingPath;
    let loadedPath;
    load3d.setBackgroundImage = (path) => {
        if (pendingPath === path) return pending;
        if (pendingPath === undefined && loadedPath === path && scene.backgroundTexture) return pending;
        const next = pending.catch(() => {}).then(async () => {
            const camera = snapshotViewerCamera(node, load3d);
            await setBackgroundImage(path);
            restoreViewerCamera(node, load3d, camera);
            loadedPath = scene.backgroundTexture ? path : undefined;
        });
        pendingPath = path;
        pending = next.finally(() => {
            if (pending === tracked) pendingPath = undefined;
        });
        const tracked = pending;
        backgroundLoads.set(load3d, pending);
        return pending;
    };
}

function getViewerController(node, modules) {
    if (!viewerControllers.has(node)) viewerControllers.set(node, modules.useLoad3d(node));
    return viewerControllers.get(node);
}

function installViewerOutputSerializer(node) {
    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    if (!imageWidget || imageWidget.__trellis2SerializeGuard) return;

    const originalSerializeValue = imageWidget.serializeValue;
    imageWidget.__trellis2SerializeGuard = true;
    imageWidget.serializeValue = async function (...args) {
        await node.__trellis2ViewerInitPromise;
        if (imageWidget.__trellis2CaptureOutputs) return imageWidget.__trellis2CaptureOutputs();
        return originalSerializeValue?.apply(this, args) ?? imageWidget.value;
    };
}

function isTemporaryPath(value) {
    return typeof value === "string" && (value.endsWith(" [temp]") || value.startsWith("temp/"));
}

async function temporaryResourceExists(value) {
    if (!isTemporaryPath(value)) return true;
    let filename = value.replace(/\s+\[temp\]$/, "").replace(/^temp\//, "");
    const slash = filename.lastIndexOf("/");
    const subfolder = slash >= 0 ? filename.slice(0, slash) : "";
    filename = slash >= 0 ? filename.slice(slash + 1) : filename;
    const query = new URLSearchParams({ filename, subfolder, type: "temp" });
    try {
        const response = await api.fetchApi(`/view?${query.toString()}`);
        return response.ok;
    } catch {
        return false;
    }
}

async function clearTemporaryViewerState(node) {
    const modelWidget = node.widgets?.find((widget) => widget.name === "model_file");
    if (modelWidget && isTemporaryPath(modelWidget.value)
        && !(await temporaryResourceExists(modelWidget.value))) {
        modelWidget.callback = null;
        modelWidget.value = "none";
    }
    node.properties ||= {};
    if (isTemporaryPath(node.properties["Last Time Model File"])
        && !(await temporaryResourceExists(node.properties["Last Time Model File"]))) {
        delete node.properties["Last Time Model File"];
        delete node.properties["Last Time Model Folder"];
    }
    const scene = node.properties["Scene Config"];
    if (scene && isTemporaryPath(scene.backgroundImage)
        && !(await temporaryResourceExists(scene.backgroundImage))) scene.backgroundImage = "";
}

function getPreviewResult(message) {
    const output = message?.output ?? message;
    return message?.ui?.result ?? output?.ui?.result ?? output?.result;
}

async function captureSceneAtQuality(load3d, width, height) {
    try {
        return await load3d.captureScene(width * CAPTURE_SCALE, height * CAPTURE_SCALE);
    } catch (error) {
        console.warn("Trellis2: supersampled capture failed; retrying at output size", error);
        return load3d.captureScene(width, height);
    }
}

async function getNativeLoad3D() {
    if (!getNativeLoad3D.promise) {
        getNativeLoad3D.promise = (async () => {
            const response = await api.fetchApi("/tbg/load3d-modules");
            if (!response.ok) throw new Error(await response.text());
            const assets = await response.json();
            return Promise.all(["useLoad3d", "Load3DConfiguration", "settingStore", "load3dSerialize"].map(
                (name) => import(new URL(`../..${assets[name]}`, import.meta.url).href)
            ));
        })().then(([useLoad3d, configuration, settingStore, load3dSerialize]) => ({
            useLoad3d: Object.values(useLoad3d).find((value) => value?.name === "useLoad3d"),
            isSceneDirty: Object.values(useLoad3d).find((value) => value?.name === "isLoad3dSceneDirty"),
            markSceneDirty: Object.values(useLoad3d).find((value) => value?.name === "markLoad3dSceneDirty"),
            getOutputCache: Object.values(useLoad3d).find((value) => value?.name === "getLoad3dOutputCache"),
            setOutputCache: Object.values(useLoad3d).find((value) => value?.name === "setLoad3dOutputCache"),
            Load3DConfiguration: Object.values(configuration).find((value) => value?.name === "Load3DConfiguration"),
            Load3dUtils: Object.values(settingStore).find((value) => typeof value?.uploadTempImage === "function"),
            snapshotLoad3dState: Object.values(load3dSerialize).find((value) => value?.name === "snapshotLoad3dState"),
        })).catch((error) => {
            getNativeLoad3D.promise = null;
            throw error;
        });
    }
    return getNativeLoad3D.promise;
}

async function saveViewerState(node) {
    const modules = await getNativeLoad3D();
    const load3d = await new Promise((resolve) => getViewerController(node, modules).waitForLoad3d(resolve));
        node.properties ||= {};
        snapshotViewerCamera(node, load3d);

        const gizmo = load3d.getGizmoTransform?.();
        if (gizmo) {
            const model = node.properties["Model Config"] || {};
            node.properties["Model Config"] = {
                ...model,
                gizmo: {
                    ...(model.gizmo || {}),
                    ...gizmo,
                },
            };
        }
}

function preserveCameraDuringModelLoads(node, load3d, modelWidget, modules) {
    const loadModel = modelWidget.callback;
    modelWidget.callback = (value) => {
        const modelFile = typeof value === "string" ? value.replaceAll("\\", "/") : value;
        if (modelFile === node.__tbgLoadedModelPath && load3d.getCurrentModel?.()) {
            return node.__tbgCameraRestorePromise || Promise.resolve();
        }
        const camera = snapshotViewerCamera(node, load3d);
        loadModel?.(value);
        const restore = load3d.whenLoadIdle().then(() => {
            restoreViewerCamera(node, load3d, camera);
            node.__tbgLoadedModelPath = load3d.getCurrentModel() ? modelFile : undefined;
            modules.markSceneDirty(node);
        });
        node.__tbgCameraRestorePromise = restore;
        return restore;
    };
}

function bindViewerOutputs(node, load3d, modules) {
    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    if (!imageWidget || imageWidget.__trellis2ViewerOutputs === load3d) return;

    let capturePromise;
    const capture = async () => {
        await node.__tbgConfigurePromise;
        await load3d.whenLoadIdle();
        await backgroundLoads.get(load3d);
        const width = node.widgets?.find((widget) => widget.name === "width")?.value || 1024;
        const height = node.widgets?.find((widget) => widget.name === "height")?.value || 1024;
        const domElement = load3d.domElement;
        if (domElement && (!domElement.clientWidth || !domElement.clientHeight)) {
            // A viewer in a collapsed node, subgraph, or inactive canvas can
            // have a zero DOM size. Capture still has to use the node output
            // dimensions, independent of whether that canvas is visible.
            load3d.setTargetSize(width, height);
            load3d.forceRender?.();
        }
        // Camera and gizmo changes must be reflected on every run.  A native
        // camera event normally marks the scene dirty, but taking the snapshot
        // here also covers changes made while a render/capture was in flight.
        const state = modules.snapshotLoad3dState(node, load3d);
        if (!state.model_3d_info?.length) {
            const savedModels = node.properties?.["Scene Config"]?.models;
            if (Array.isArray(savedModels) && savedModels.length) {
                state.model_3d_info = savedModels;
            }
        }
        const cached = modules.getOutputCache(node);
        if (!modules.isSceneDirty(node) && cached
            && JSON.stringify(cached.camera_info) === JSON.stringify(state.camera_info)
            && JSON.stringify(cached.model_3d_info) === JSON.stringify(state.model_3d_info)) return cached;
        const connectedFile = node.inputs?.some((input) => input.name === "model_3d" && input.link != null);
        if (connectedFile && load3d.isSplatModel()) {
            return {
                image: "", mask: "", normal: "",
                camera_info: state.camera_info,
                recording: "",
                model_3d_info: state.model_3d_info,
            };
        }
        const captured = await captureSceneAtQuality(load3d, width, height);
        const [scene, mask, normal] = await Promise.all([
            modules.Load3dUtils.uploadTempImage(captured.scene, "scene"),
            modules.Load3dUtils.uploadTempImage(captured.mask, "scene_mask"),
            modules.Load3dUtils.uploadTempImage(captured.normal, "scene_normal"),
        ]);

        load3d.handleResize();
        const result = {
            image: `threed/${scene.name} [temp]`,
            mask: `threed/${mask.name} [temp]`,
            normal: `threed/${normal.name} [temp]`,
            camera_info: state.camera_info,
            recording: "",
            model_3d_info: state.model_3d_info,
        };
        modules.setOutputCache(node, result);
        return result;
    };
    const serializeCapture = () => {
        capturePromise ||= capture().finally(() => { capturePromise = null; });
        return capturePromise;
    };
    imageWidget.__trellis2CaptureOutputs = serializeCapture;
    if (!imageWidget.__trellis2SerializeGuard) imageWidget.serializeValue = serializeCapture;
    imageWidget.__trellis2ViewerOutputs = load3d;
}

async function configureNativeViewerNow(node, loadFolder = "input", preview = null) {
    const inputModelWidget = node.widgets?.find((widget) => widget.name === "model_file");
    if (!inputModelWidget) return;

    const modules = await getNativeLoad3D();
    const { Load3DConfiguration } = modules;
    const load3d = await new Promise((resolve) => getViewerController(node, modules).waitForLoad3d(resolve));
    disableViewerInertia(load3d);
    guardViewerBackground(node, load3d);
    node.__tbgViewerContainer = load3d.domElement?.parentElement;
    guardViewerEvents(node.__tbgViewerContainer);
    bindViewerOutputs(node, load3d, modules);
    if (preview?.size) {
        for (const name of ["width", "height"]) {
            const widget = node.widgets?.find((item) => item.name === name);
            if (widget && widget.value !== preview.size[name]) widget.value = preview.size[name];
        }
    }
        const width = node.widgets?.find((widget) => widget.name === "width");
        const height = node.widgets?.find((widget) => widget.name === "height");
        const modelWidget = inputModelWidget;
        const modelFile = preview?.modelFile || modelWidget.value;
        if (modelFile && modelFile !== "none" && Array.isArray(modelWidget.options?.values)
            && !modelWidget.options.values.includes(modelFile)) {
            modelWidget.options.values.push(modelFile);
        }

        if (node.__tbgConfiguredViewer === load3d) {
            await load3d.whenLoadIdle();
            const sizeKey = `${width.value}x${height.value}`;
            if (node.__tbgTargetSize !== sizeKey) {
                load3d.setTargetSize(width.value, height.value);
                node.__tbgTargetSize = sizeKey;
                modules.markSceneDirty(node);
            }
            const modelMissing = modelFile && modelFile !== "none" && !load3d.getCurrentModel?.();
            if (modelFile && modelFile !== "none"
                && (modelFile !== node.__tbgLoadedModelPath || modelMissing)) {
                await saveViewerState(node);
                modelWidget.value = modelFile;
                await (node.__tbgCameraRestorePromise || load3d.whenLoadIdle());
                load3d.forceRender?.();
                node.__tbgLoadedModelPath = load3d.getCurrentModel() ? modelFile : undefined;
                modules.markSceneDirty(node);
            }
            const backgroundMissing = !load3d.getSceneManager?.().backgroundTexture;
            if (preview?.backgroundPath
                && (preview.backgroundPath !== node.__tbgBackgroundPath || backgroundMissing)) {
                load3d.setBackgroundRenderMode("tiled");
                await load3d.setBackgroundImage(preview.backgroundPath);
                const scene = load3d.getSceneManager();
                if ((!load3d.domElement?.clientWidth || !load3d.domElement?.clientHeight)
                    && scene.backgroundTexture && scene.backgroundMesh) {
                    scene.updateBackgroundSize(scene.backgroundTexture, scene.backgroundMesh,
                        width.value, height.value);
                }
                node.__tbgBackgroundPath = load3d.getSceneManager().backgroundTexture ? preview.backgroundPath : undefined;
                modules.markSceneDirty(node);
            }
            load3d.toggleGrid(false);
            return;
        }

        // Load3DConfiguration chains the previous widget callback. Trellis2
        // changes between input files and generated output GLBs, so replace the
        // callback rather than leaving an input loader attached to an output file.
        modelWidget.callback = null;
        modelWidget.value = "none";
        node.properties ||= {};
        (node.properties["Scene Config"] ||= {}).showGrid = false;
        new Load3DConfiguration(load3d, node.properties).configure({
            loadFolder,
            modelWidget,
            cameraState: preview?.cameraInfo,
            width,
            height,
            bgImagePath: preview?.backgroundPath,
            silentOnNotFound: true,
            onSceneInvalidated: () => {
                modules.markSceneDirty(node);
                node.setDirtyCanvas?.(true, true);
            },
        });
        preserveCameraDuringModelLoads(node, load3d, modelWidget, modules);
        if (modelFile && modelFile !== "none") {
            modelWidget.value = modelFile;
            await (node.__tbgCameraRestorePromise || load3d.whenLoadIdle());
            node.__tbgLoadedModelPath = load3d.getCurrentModel() ? modelFile : undefined;
        }
        const backgroundPath = preview?.backgroundPath || node.properties["Scene Config"]?.backgroundImage;
        if (backgroundPath) {
            await load3d.setBackgroundImage(backgroundPath);
            const scene = load3d.getSceneManager();
            if ((!load3d.domElement?.clientWidth || !load3d.domElement?.clientHeight)
                && scene.backgroundTexture && scene.backgroundMesh) {
                scene.updateBackgroundSize(scene.backgroundTexture, scene.backgroundMesh,
                    width.value, height.value);
            }
        }
        node.__tbgTargetSize = `${width.value}x${height.value}`;
        node.__tbgBackgroundPath = node.properties["Scene Config"]?.backgroundImage;
        node.__tbgConfiguredViewer = load3d;
        modules.markSceneDirty(node);
}

async function configureNativeViewer(node, loadFolder = "input", preview = null) {
    const previous = node.__tbgConfigurePromise || Promise.resolve();
    let release;
    const current = new Promise((resolve) => { release = resolve; });
    const queued = previous.then(() => current);
    node.__tbgConfigurePromise = queued;
    await previous;
    try {
        await configureNativeViewerNow(node, loadFolder, preview);
    } finally {
        release();
        if (node.__tbgConfigurePromise === queued) node.__tbgConfigurePromise = null;
    }
}

async function applyPreviewResult(node, message) {
    const result = getPreviewResult(message);
    if (!Array.isArray(result) || typeof result[0] !== "string") return;

    if (!node.__tbgConfiguredViewer) await initializeNativeViewer(node);

    const modelWidget = node.widgets?.find((widget) => widget.name === "model_file");
    if (!modelWidget) return;

    const modelFile = result[0].replaceAll("\\", "/");
    node.properties ||= {};
    const backgroundPath = Array.isArray(result[2]) ? result[2][0] : result[2];
    await configureNativeViewer(node, "input", { modelFile: modelFile || modelWidget.value, backgroundPath, size: result[3] });
    node.properties["Last Time Model File"] = modelFile;
    node.properties["Last Time Model Folder"] = "input";
    node.setDirtyCanvas?.(true, true);
}

function findNode(nodeId) {
    const root = app.rootGraph || app.graph;
    if (!root) return null;
    const id = String(nodeId);
    const direct = root.getNodeById?.(Number(id)) || root.getNodeById?.(id);
    if (direct) return direct;

    // Execution updates for nodes inside subgraphs use a locator such as
    // "subgraph-uuid:local-node-id". Resolve that against the same graph
    // hierarchy used by the native Load3D extension.
    const separator = id.lastIndexOf(":");
    if (separator > 0) {
        const subgraphId = id.slice(0, separator);
        const localId = id.slice(separator + 1);
        const subgraph = root.subgraphs?.get?.(subgraphId);
        const nested = subgraph?.getNodeById?.(Number(localId)) || subgraph?.getNodeById?.(localId);
        if (nested) return nested;
    }

    const visit = (graph) => {
        for (const candidate of graph.nodes || graph._nodes || []) {
            if (candidate.isSubgraphNode?.() && candidate.subgraph) {
                const nested = visit(candidate.subgraph);
                if (nested) return nested;
            }
        }
        return graph.getNodeById?.(Number(id)) || graph.getNodeById?.(id) || null;
    };
    return visit(root);
}

function isTrellisLoad3D(node) {
    return node?.constructor?.trellis2Load3D === true;
}

function initializeNativeViewer(node) {
    if (node.__trellis2ViewerInitPromise) return node.__trellis2ViewerInitPromise;
    node.__trellis2ViewerInitPromise = (async () => {
        if (!node.__trellis2StaleStateCleared) {
            await clearTemporaryViewerState(node);
            node.__trellis2StaleStateCleared = true;
        }
        await configureNativeViewer(node);
    })().catch((error) => {
        console.error("Trellis2: failed to initialize 3D viewer", error);
    }).finally(() => {
        node.__trellis2ViewerInitPromise = null;
    });
    return node.__trellis2ViewerInitPromise;
}

app.registerExtension({
    name: "Trellis2.Load3D",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TBG_NODE_ID
            && nodeData.name !== TBG_DISPLAY_NAME
            && nodeData.display_name !== TBG_DISPLAY_NAME) return;

        nodeType.trellis2Load3D = true;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onNodeCreated = function () {
            const output = originalOnNodeCreated?.apply(this, arguments);
            installViewerOutputSerializer(this);
            initializeNativeViewer(this);
            return output;
        };
        nodeType.prototype.onExecuted = function (message) {
            const output = originalOnExecuted?.apply(this, arguments);
            applyPreviewResult(this, message).catch((error) => console.error("Trellis2: failed to update 3D viewer", error));
            return output;
        };
    },

    onNodeOutputsUpdated(outputs) {
        for (const [nodeId, output] of Object.entries(outputs || {})) {
            const node = findNode(nodeId);
            if (!isTrellisLoad3D(node)) continue;
            applyPreviewResult(node, output).catch((error) => console.error("Trellis2: failed to update 3D viewer", error));
        }
    },

    loadedGraphNode(node) {
        if (isTrellisLoad3D(node)) {
            installViewerOutputSerializer(node);
            initializeNativeViewer(node);
        }
    },
});
