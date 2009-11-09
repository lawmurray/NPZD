--
-- SQLite database initialisation script.
--
-- @author Lawrence Murray <lawrence.murray@csiro.au>
-- $Rev$
-- $Date$
--

CREATE TABLE IF NOT EXISTS Trait (
  Name VARCHAR PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Category (
  Name VARCHAR PRIMARY KEY,
  Description VARCHAR,
  Position INTEGER UNIQUE
);

CREATE TABLE IF NOT EXISTS Node (
  Name VARCHAR PRIMARY KEY,
  LaTeXName VARCHAR,
  Description VARCHAR,
  Category VARCHAR NOT NULL REFERENCES Category(Name),
  Position INTEGER
);

CREATE TABLE IF NOT EXISTS Edge (
  ParentNode VARCHAR NOT NULL REFERENCES Node(Name),
  ChildNode VARCHAR NOT NULL REFERENCES Node(Name),
  Position INTEGER,
  PRIMARY KEY (ParentNode, ChildNode),
  UNIQUE (ChildNode, Position)
);

CREATE TABLE IF NOT EXISTS NodeTrait (
  Node VARCHAR NOT NULL REFERENCES Node(Name),
  Trait VARCHAR NOT NULL REFERENCES Trait(Name),
  PRIMARY KEY (Node, Trait)
);

CREATE TABLE IF NOT EXISTS NodeFormula (
  Node VARCHAR NOT NULL REFERENCES Node(Name),
  Function VARCHAR NOT NULL,
  Formula VARCHAR NOT NULL,
  LaTeXFormula VARCHAR,
  PRIMARY KEY (Node, Function)
);

--
-- Foreign keys
--

-- Node.Category -> NodeCategory.Name
CREATE TRIGGER IF NOT EXISTS NodeInsert AFTER INSERT ON Node
  WHEN
    (SELECT 1 FROM Category WHERE Name = NEW.Category) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Category does not exist');
  END;
    
-- NodeTrait.Node -> Node.Name
CREATE TRIGGER IF NOT EXISTS NodeTraitNodeInsert AFTER INSERT ON NodeTrait
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.Node) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Node does not exist');
  END;

-- NodeTrait.Trait -> Trait.Trait
CREATE TRIGGER IF NOT EXISTS NodeTraitTraitInsert AFTER INSERT ON NodeTrait
  WHEN
    (SELECT 1 FROM Trait WHERE Name = NEW.Trait) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Trait does not exist');
  END;

-- Edge.ParentName -> Node.Name
CREATE TRIGGER IF NOT EXISTS EdgeParentNodeInsert AFTER INSERT ON Edge
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.ParentNode) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Dependency does not exist');
  END;

-- Edge.ChildNode -> Node.Name
CREATE TRIGGER IF NOT EXISTS EdgeChildNodeInsert AFTER INSERT ON Edge
  WHEN
    (SELECT 1 FROM Node WHERE Name = NEW.ChildNode) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Node does not exist');
  END;

-- Cascades
CREATE TRIGGER IF NOT EXISTS NodeUpdate
  AFTER
    UPDATE OF Name ON Node
  BEGIN
    UPDATE NodeTrait SET Node = NEW.Name WHERE Node = OLD.Name;
    UPDATE Edge SET ParentNode = NEW.Name WHERE ParentNode = OLD.Name;
    UPDATE Edge SET ChildNode = NEW.Name WHERE ChildNode = OLD.Name;
  END;

CREATE TRIGGER IF NOT EXISTS NodeDelete
  AFTER
    DELETE ON Node
  BEGIN
    DELETE FROM NodeTrait WHERE Node = OLD.Name;
    DELETE FROM Edge WHERE ParentNode = OLD.Name OR ChildNode = OLD.Name;
  END;

CREATE TRIGGER IF NOT EXISTS TraitUpdate
  AFTER
    UPDATE OF Name ON Trait
  BEGIN
    UPDATE NodeTrait SET Trait = NEW.Name WHERE Trait = OLD.Name;
  END;

CREATE TRIGGER IF NOT EXISTS TraitDelete
  AFTER
    DELETE ON Trait
  BEGIN
    DELETE FROM NodeTrait WHERE Trait = OLD.Name;
  END;

--
-- Other constraints
--
CREATE TRIGGER IF NOT EXISTS FormulaCheck AFTER INSERT ON Edge
  WHEN
    (SELECT 1 FROM Node, NodeFormula WHERE Node.Name = NEW.ChildNode AND
    NodeFormula.Formula LIKE '%' || NEW.ParentNode || '%' AND
    NodeFormula.Node = Node.Name) IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Dependency is not mentioned in formula');
  END;

--
-- Clear tables (in case they already existed)
--
DELETE FROM Node;
DELETE FROM Trait;
DELETE FROM Category;
DELETE FROM NodeTrait;
DELETE FROM NodeFormula;
DELETE FROM Edge;

--
-- Populate Trait
--
INSERT INTO Trait VALUES ('IS_S_NODE');
INSERT INTO Trait VALUES ('IS_D_NODE');
INSERT INTO Trait VALUES ('IS_C_NODE');
INSERT INTO Trait VALUES ('IS_R_NODE');
INSERT INTO Trait VALUES ('IS_F_NODE');
INSERT INTO Trait VALUES ('IS_O_NODE');
INSERT INTO Trait VALUES ('IS_P_NODE');
INSERT INTO Trait VALUES ('IS_GENERIC_STATIC');
INSERT INTO Trait VALUES ('IS_GENERIC_FORWARD');
INSERT INTO Trait VALUES ('IS_ODE_FORWARD');
INSERT INTO Trait VALUES ('IS_UNIFORM_VARIATE');
INSERT INTO Trait VALUES ('IS_GAUSSIAN_VARIATE');
INSERT INTO Trait VALUES ('IS_NORMAL_VARIATE');
INSERT INTO Trait VALUES ('IS_GAUSSIAN_LIKELIHOOD');
INSERT INTO Trait VALUES ('IS_NORMAL_LIKELIHOOD');
INSERT INTO Trait VALUES ('IS_LOG_NORMAL_LIKELIHOOD');
INSERT INTO Trait VALUES ('HAS_ZERO_MU');
INSERT INTO Trait VALUES ('HAS_UNIT_SIGMA');
INSERT INTO Trait VALUES ('HAS_COMMON_SIGMA');
INSERT INTO Trait VALUES ('HAS_GAUSSIAN_PRIOR');
INSERT INTO Trait VALUES ('HAS_LOG_NORMAL_PRIOR');

--
-- Populate category
--
INSERT INTO Category VALUES('Constant', '', 1);
INSERT INTO Category VALUES('Parameter', '', 2);
INSERT INTO Category VALUES('Forcing', '', 3);
INSERT INTO Category VALUES('Observation', '', 4);
INSERT INTO Category VALUES('Random variate', 'Representing pseudorandom variates required in the update of other variables.', 5);
INSERT INTO Category VALUES('Static variable', '.', 6);
INSERT INTO Category VALUES('Discrete-time variable', '', 7);
INSERT INTO Category VALUES('Continuous-time variable', '', 8);
INSERT INTO Category VALUES('Intermediate result', 'Representing intermediate evaluations which may be reused multiple times for convenience. Will be inlined.', 9);

