const tasks = require('./routes/tasks)
app.use(tasks.routes(), tasks.allowedMethods())
