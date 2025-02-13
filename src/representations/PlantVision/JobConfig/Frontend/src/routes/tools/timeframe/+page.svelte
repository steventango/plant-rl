<!-- App.svelte -->
<script lang="ts">
    // import { createEventDispatcher } from 'svelte';
    import DownloadJson from '../../../components/download-json.svelte';
    
    let timeframe: { date: string, ontime: string, offtime: string }[] = [];
    // const dispatch = createEventDispatcher();

    let ontime = '';
    let offtime = '';
    let startday = '';

    function addTimeframe() {
        const newTimeframe = { date: startday, ontime, offtime };
        timeframe = [...timeframe, newTimeframe];
        // Reset input fields after saving
        ontime = '';
        offtime = '';
        startday = '';
    }

    function deleteTimeframe(index) {
        timeframe = timeframe.filter((_, i) => i !== index);
    }
</script>

<div class="grid grid-cols-2">
    <form on:submit|preventDefault={addTimeframe}>
        <label>
            <span>Enter info<br /></span>
            <input type="date" bind:value={startday} />
            <input type="time" bind:value={ontime} />
            to
            <input type="time" bind:value={offtime} />
            <button class="btn variant-filled-primary"type="submit">Save</button>
        </label>
        
    </form>

    <div class="flex flex-col">
        {#each timeframe as tf, index}
            <div class="mb-4">
                <p>Date: {tf.date}</p>
                <p>Start time: {tf.ontime}</p>
                <p>End time: {tf.offtime}</p>
                <button class="btn variant-filled-error" on:click={() => deleteTimeframe(index)}>Delete</button>
            </div>
        {/each}
    </div>
</div>

<DownloadJson
filename="timeframe.json"
dataObjToWrite={timeframe}
/>
