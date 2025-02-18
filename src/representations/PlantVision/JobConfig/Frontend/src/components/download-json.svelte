<script lang="ts">

    export let filename: string;
    export let dataObjToWrite: Object;
    export let onclickprep;
    export let onclickpreparg;

const saveTemplateAsFile = (filename: string, dataObjToWrite: Object) => {
    const blob = new Blob([JSON.stringify(dataObjToWrite)], { type: "text/json" });
    const link = document.createElement("a");

    link.download = filename;
    link.href = window.URL.createObjectURL(blob);
    link.dataset.downloadurl = ["text/json", link.download, link.href].join(":");

    const evt = new MouseEvent("click", {
        view: window,
        bubbles: true,
        cancelable: true,
    });

    link.dispatchEvent(evt);
    link.remove()
};

</script>


<!-- download button -->
<button class="btn variant-filled-primary" on:click={() => {
    if ((onclickprep != null) && (onclickpreparg != null)) {
        dataObjToWrite = onclickprep(onclickpreparg);
    }
    saveTemplateAsFile(filename, dataObjToWrite)}}>Download</button>