import { createStore } from 'vuex'
import { getAPI } from './axios-api'

export default createStore({
  state: {
     accessToken: null,
     refreshToken: null,
     APIData: ''
  },
  mutations: {
    updateStorage (state, { access, refresh }) {
      state.accessToken = access
      state.refreshToken = refresh
    },
    destroyToken (state) {
      state.accessToken = null
      state.refreshToken = null
    }
  },
  getters: {
    loggedIn (state) {
      return state.accessToken != null
    }
  },
  actions: {
    userLogout (context) {
      if (context.getters.loggedIn) {
          context.commit('destroyToken')
      }
    },
    userLogin (context, usercredentials) {
      return new Promise((resolve, reject) => {
        getAPI.post('/api-token/', {
          username: usercredentials.username,
          password: usercredentials.password
        })
          .then(response => {
            context.commit('updateStorage', { access: response.data.access, refresh: response.data.refresh }) 
            resolve()
          })
          .catch(err => {
            reject(err)
          })
      })
    }
  }
})