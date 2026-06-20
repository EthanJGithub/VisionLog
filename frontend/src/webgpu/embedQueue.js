import { embedImage } from "./clip";
import { api } from "../api";

/**
 * Background CLIP embedder for object crops. Crops are queued at capture time and embedded +
 * uploaded in small batches on a timer — entirely OFF the detection loop, so it never costs FPS.
 * Keeps draining after detection stops until the queue empties (so late captures still get
 * embedded). All best-effort: embedding/upload failures are swallowed (search degrades to class
 * filtering).
 *
 * @param getSourceId  () => current logging source id (or null)
 * @param isRunning    () => whether detection is still active
 */
export function createEmbedder(getSourceId, isRunning, { batch = 4, intervalMs = 1500 } = {}) {
  let queue = [];
  let active = false;

  async function tick() {
    if (!queue.length) return;
    const take = queue.splice(0, batch);
    const items = [];
    for (const { track_id, thumb } of take) {
      const embedding = await embedImage(thumb);
      if (embedding) items.push({ track_id, embedding });
    }
    const sid = getSourceId();
    if (items.length && sid != null) {
      try { await api.setEmbeddings(sid, items); } catch { /* best-effort */ }
    }
  }

  function pump() {
    const loop = async () => {
      await tick();
      if (!isRunning() && !queue.length) { active = false; return; }
      setTimeout(loop, intervalMs);
    };
    setTimeout(loop, intervalMs);
  }

  return {
    enqueue(track_id, thumb) {
      queue.push({ track_id, thumb });
      if (!active) { active = true; pump(); }
    },
  };
}
