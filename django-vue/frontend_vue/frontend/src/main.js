import { createApp } from 'vue'
import App from './App.vue'
import router from './routes.js'
import store from './store'
import 'bootstrap/dist/css/bootstrap.min.css'
import VueMobileDetection from 'vue-mobile-detection'


createApp(App).use(VueMobileDetection).use(store).use(router).mount('#app')

router.beforeEach((to, from, next) => {
    if (to.matched.some(record => record.meta.requiresLogin)) {
      if (!store.getters.loggedIn) {
        next({ name: 'log-in' })
      } else {
        next()
      }
    } else {
      next()
    }
  })