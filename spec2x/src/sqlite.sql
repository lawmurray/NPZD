--
-- SQLite database initialisation script.
--
-- @author Lawrence Murray <lawrence.murray@csiro.au>
-- $Rev$
-- $Date$
--

CREATE TABLE IF NOT EXISTS Node (
  Name VARCHAR PRIMARY KEY,
  LaTeXName VARCHAR,
  Formula VARCHAR,
  LaTeXFormula VARCHAR,
  Description VARCHAR,
  Type VARCHAR NOT NULL,
  Position INTEGER
);

CREATE TABLE IF NOT EXISTS Trait (
  Name VARCHAR PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS NodeTrait (
  Node VARCHAR NOT NULL REFERENCES Node(Name),
  Trait VARCHAR NOT NULL REFERENCES Trait(Name),
  PRIMARY KEY (Node, Trait)
);

CREATE TABLE IF NOT EXISTS Edge (
  ParentNode VARCHAR NOT NULL REFERENCES Node(Name),
  ChildNode VARCHAR NOT NULL REFERENCES Node(Name),
  Position INTEGER,
  PRIMARY KEY (ParentNode, ChildNode),
  UNIQUE (ChildNode, Position)
);

--
-- Foreign keys
--

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
    (SELECT 1 FROM Node, Edge WHERE Name = NEW.ChildNode AND Formula LIKE
    '%' || NEW.ParentNode || '%') IS NULL
  BEGIN
    SELECT RAISE(ABORT, 'Dependency is not mentioned in formula');
  END;


--
-- Clear tables (in case they already existed)
--
DELETE FROM Node;
DELETE FROM Trait;
DELETE FROM NodeTrait;
DELETE FROM Edge;

--
-- Populate Trait
--
INSERT INTO Trait VALUES ('IS_IN_NODE');
INSERT INTO Trait VALUES ('IS_EX_NODE');
INSERT INTO Trait VALUES ('IS_R_NODE');
INSERT INTO Trait VALUES ('IS_F_NODE');
INSERT INTO Trait VALUES ('IS_GENERIC_STATIC');
INSERT INTO Trait VALUES ('IS_GENERIC_FORWARD');
INSERT INTO Trait VALUES ('IS_ODE_FORWARD');
INSERT INTO Trait VALUES ('IS_UNIFORM');
INSERT INTO Trait VALUES ('IS_GAUSSIAN');
