<template>
  <div class="log">
    
  </div>
</template>

<script>
  export default {
    created () {
      this.$store.dispatch('userLogout')
        .then(() => {
          this.$router.push({ name: 'log-in' })
        })
    }
  }
</script>

<style>
</style>